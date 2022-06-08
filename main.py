import argparse
import copy
import csv
import os
import warnings

import numpy
import torch
import tqdm
from timm import utils
from torch.utils import data
from torchvision import transforms

from nets import nn
from utils import util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")
data_dir = os.path.join('..', 'Dataset', 'IMAGENET')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def lr(args):
    return 0.256 * args.batch_size * args.world_size / 4096


def batch(samples, targets, model, criterion):
    samples = samples.cuda()
    targets = targets.cuda()

    with torch.cuda.amp.autocast():
        outputs = model(samples)

    acc1, acc5 = util.accuracy(outputs, targets, top_k=(1, 5))
    return criterion(outputs, targets), acc1, acc5


def train(args):
    # progressive training params
    total_step = 4
    drop_rates = numpy.linspace(0, .2, total_step)
    magnitudes = numpy.linspace(5, 10, total_step)

    model = nn.EfficientNet().cuda()
    ema_m = nn.EMA(model)

    amp_scale = torch.cuda.amp.GradScaler()
    optimizer = nn.RMSprop(util.weight_decay(model), lr(args))
    if not args.distributed:
        model = torch.nn.parallel.DataParallel(model)
    else:
        model = torch.nn.parallel.DistributedDataParallel(model, [args.local_rank])

    scheduler = nn.StepLR(optimizer)
    if args.poly:
        criterion = nn.PolyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    with open(f'weights/step.csv', 'w') as f:
        best = 0
        if args.local_rank == 0:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'acc@1', 'acc@5', 'train_loss', 'val_loss'])
            writer.writeheader()
        for step in range(total_step):
            model.module.drop_rate = drop_rates[step]
            ratio = float(step + 1) / total_step
            start_epoch = int(float(step) / total_step * args.epochs)
            end_epoch = int(ratio * args.epochs)
            input_size = int(128 + (args.input_size - 128) * ratio)

            sampler = None
            dataset = Dataset(os.path.join(data_dir, 'train'),
                              transforms.Compose([util.Resize(input_size),
                                                  util.RandomAugment(magnitudes[step]),
                                                  transforms.RandomHorizontalFlip(0.5),
                                                  transforms.ToTensor(), normalize]))
            if args.distributed:
                sampler = data.distributed.DistributedSampler(dataset)

            loader = data.DataLoader(dataset, args.batch_size, not args.distributed,
                                     sampler=sampler, num_workers=8, pin_memory=True)

            for epoch in range(start_epoch, end_epoch):
                if args.distributed:
                    sampler.set_epoch(epoch)
                p_bar = loader
                if args.local_rank == 0:
                    print(('\n' + '%10s' * 3) % ('epoch', 'memory', 'loss'))
                    p_bar = tqdm.tqdm(loader, total=len(loader))
                model.train()
                m_loss = util.AverageMeter()
                for samples, targets in p_bar:
                    loss, _, _ = batch(samples, targets, model, criterion)

                    optimizer.zero_grad()

                    amp_scale.scale(loss).backward()
                    amp_scale.step(optimizer)
                    amp_scale.update()

                    ema_m.update(model)
                    torch.cuda.synchronize()

                    if args.distributed:
                        loss = utils.reduce_tensor(loss.data, args.world_size)

                    m_loss.update(loss.item(), samples.size(0))
                    if args.local_rank == 0:
                        gpus = '%.4gG' % (torch.cuda.memory_reserved() / 1E9)
                        desc = ('%10s' * 2 + '%10.3g') % ('%g/%g' % (epoch + 1, args.epochs), gpus, m_loss.avg)
                        p_bar.set_description(desc)

                scheduler.step(epoch + 1)
                if args.local_rank == 0:
                    val_loss, acc1, acc5 = test(ema_m.model.eval())
                    writer.writerow({'acc@1': str(f'{acc1:.3f}'),
                                     'acc@5': str(f'{acc5:.3f}'),
                                     'epoch': str(epoch + 1).zfill(3),
                                     'val_loss': str(f'{val_loss:.3f}'),
                                     'train_loss': str(f'{m_loss.avg:.3f}')})
                    state = {'model': copy.deepcopy(ema_m.model).half()}
                    torch.save(state, f'weights/last.pt')
                    if acc1 > best:
                        torch.save(state, f'weights/best.pt')
                    best = max(acc1, best)

                    del state

            del dataset
            del sampler
            del loader

    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


def test(model=None):
    if model is None:
        model = torch.load('weights/best.pt', map_location='cuda')['model'].float()
        model.eval()

    criterion = torch.nn.CrossEntropyLoss().cuda()

    dataset = Dataset(os.path.join(data_dir, 'val'),
                      transforms.Compose([transforms.Resize(384),
                                          transforms.CenterCrop(384),
                                          transforms.ToTensor(), normalize]))

    loader = data.DataLoader(dataset, 32, num_workers=8, pin_memory=True)

    top1 = util.AverageMeter()
    top5 = util.AverageMeter()

    m_loss = util.AverageMeter()

    with torch.no_grad():
        for samples, targets in tqdm.tqdm(loader, ('%10s' * 2) % ('acc@1', 'acc@5')):
            loss, acc1, acc5 = batch(samples, targets, model, criterion)

            torch.cuda.synchronize()

            top1.update(acc1.item(), samples.size(0))
            top5.update(acc5.item(), samples.size(0))

            m_loss.update(loss.item(), samples.size(0))
        acc1, acc5 = top1.avg, top5.avg
        print('%10.3g' * 2 % (acc1, acc5))
    if model is None:
        torch.cuda.empty_cache()
    else:
        return m_loss.avg, acc1, acc5


def profile(args):
    model = nn.EfficientNet().export()
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    if args.local_rank == 0:
        print(f'Number of parameters: {int(params)}')
        if args.benchmark:
            util.print_benchmark(model, (1, 3, 384, 384))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--input-size', default=300, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--poly', action='store_true')

    args = parser.parse_args()

    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    utils.random_seed(0, args.local_rank)

    profile(args)

    if args.train:
        train(args)
    if args.test:
        test()


if __name__ == '__main__':
    main()
