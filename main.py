import argparse
import copy
import csv
import os
import random
import warnings

import numpy
import torch
import tqdm
from torch.utils import data
from torchvision import transforms

from nets import nn
from utils import util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")
data_dir = os.path.join('..', 'Dataset', 'IMAGENET')


def set_seed():
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def lr(args):
    lr_base = 0.256 / 4096
    if not args.distributed:
        return args.batch_size * lr_base
    else:
        return args.batch_size * lr_base * args.world_size


def batch(images, target, model, criterion):
    images = images.cuda()
    target = target.cuda()

    with torch.cuda.amp.autocast():
        output = model(images)

    acc1, acc5 = util.accuracy(output, target, top_k=(1, 5))
    return criterion(output, target), acc1, acc5


def train(args):
    # progressive training params
    total_step = 4
    drop_rates = numpy.linspace(0, .2, total_step)
    magnitudes = numpy.linspace(5, 10, total_step)

    model = nn.EfficientNet().cuda()
    ema_m = nn.EMA(model)

    amp_scale = torch.cuda.amp.GradScaler()
    optimizer = nn.RMSprop(util.add_weight_decay(model), lr(args), 0.9, 1e-3, 0, 0.9)
    if not args.distributed:
        model = torch.nn.parallel.DataParallel(model)
    else:
        model = torch.nn.parallel.DistributedDataParallel(model, [args.local_rank])

    scheduler = nn.StepLR(optimizer)
    criterion = nn.CrossEntropyLoss().cuda()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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
                    print(('\n' + '%10s' * 2) % ('epoch', 'loss'))
                    p_bar = tqdm.tqdm(loader, total=len(loader))
                model.train()
                train_loss = util.AverageMeter()
                for images, target in p_bar:
                    loss, _, _ = batch(images, target, model, criterion)

                    optimizer.zero_grad()
                    amp_scale.scale(loss).backward()
                    amp_scale.step(optimizer)
                    amp_scale.update()

                    ema_m.update(model)
                    torch.cuda.synchronize()

                    if args.distributed:
                        loss = loss.data.clone()
                        torch.distributed.all_reduce(loss)
                        loss /= args.world_size

                    loss = loss.item()
                    train_loss.update(loss, images.size(0))
                    if args.local_rank == 0:
                        desc = ('%10s' + '%10.3g') % ('%g/%g' % (epoch + 1, args.epochs), loss)
                        p_bar.set_description(desc)

                scheduler.step(epoch + 1)
                if args.local_rank == 0:
                    val_loss, acc1, acc5 = test(ema_m.model.eval())
                    writer.writerow({'acc@1': str(f'{acc1:.3f}'),
                                     'acc@5': str(f'{acc5:.3f}'),
                                     'epoch': str(epoch + 1).zfill(3),
                                     'val_loss': str(f'{val_loss.avg:.3f}'),
                                     'train_loss': str(f'{train_loss.avg:.3f}')})
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
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset = Dataset(os.path.join(data_dir, 'val'),
                      transforms.Compose([transforms.Resize(384),
                                          transforms.CenterCrop(384),
                                          transforms.ToTensor(), normalize]))

    loader = data.DataLoader(dataset, 32, num_workers=8, pin_memory=True)

    top1 = util.AverageMeter()
    top5 = util.AverageMeter()

    val_loss = util.AverageMeter()

    with torch.no_grad():
        for images, target in tqdm.tqdm(loader, ('%10s' * 2) % ('acc@1', 'acc@5')):
            loss, acc1, acc5 = batch(images, target, model, criterion)

            torch.cuda.synchronize()

            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            val_loss.update(loss.item(), images.size(0))
        acc1, acc5 = top1.avg, top5.avg
        print('%10.3g' * 2 % (acc1, acc5))
    if model is None:
        torch.cuda.empty_cache()
    else:
        return val_loss, acc1, acc5


def profile(args):
    model = nn.EfficientNet().export()
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    if args.local_rank == 0:
        print(f'Number of parameters: {int(params)}')
        if args.benchmark:
            util.print_benchmark(model, (1, 3, 384, 384))


def main():
    set_seed()

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--input-size', default=300, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.world_size = int(os.getenv('WORLD_SIZE', 1))
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')
    profile(args)
    if args.train:
        train(args)
    if args.test:
        test()


if __name__ == '__main__':
    main()
