import math
import random

import cv2
import numpy
import torch
from PIL import Image, ImageOps, ImageEnhance

max_value = 10.


def copy_weights(model1, model2):
    model1.eval()
    model2.eval()
    with torch.no_grad():
        m1_std = model1.state_dict().values()
        m2_std = model2.state_dict().values()
        for m1, m2 in zip(m1_std, m2_std):
            m1.copy_(m2)

    state = {'model': model1.half()}
    torch.save(state, f'weights/best_tf.pt')


def set_seed():
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def print_benchmark(model, shape):
    import os
    import onnx
    from onnx import optimizer
    from caffe2.proto import caffe2_pb2
    from caffe2.python.onnx.backend import Caffe2Backend
    from caffe2.python import core, model_helper, workspace

    inputs = torch.randn(shape, requires_grad=True)
    model(inputs)

    # export torch to onnx
    dynamic_axes = {'input0': {0: 'batch'}, 'output0': {0: 'batch'}}

    _ = torch.onnx.export(model, inputs, 'weights/model.onnx', True, False,
                          input_names=["input0"],
                          output_names=["output0"],
                          keep_initializers_as_inputs=True,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                          dynamic_axes=dynamic_axes,
                          opset_version=10)

    onnx.checker.check_model(onnx.load('weights/model.onnx'))

    # export onnx to caffe2
    onnx_model = onnx.load('weights/model.onnx')

    # Optimizer passes to perform
    passes = ['eliminate_identity',
              'eliminate_deadend',
              'eliminate_nop_dropout',
              'eliminate_nop_pad',
              'eliminate_nop_transpose',
              'eliminate_unused_initializer',
              'extract_constant_to_initializer',
              'fuse_add_bias_into_conv',
              'fuse_bn_into_conv',
              'fuse_consecutive_concats',
              'fuse_consecutive_reduce_unsqueeze',
              'fuse_consecutive_squeezes',
              'fuse_consecutive_transposes',
              'fuse_matmul_add_bias_into_gemm',
              'fuse_transpose_into_gemm',
              'lift_lexical_references',
              'fuse_pad_into_conv']
    onnx_model = optimizer.optimize(onnx_model, passes)
    caffe2_init, caffe2_predict = Caffe2Backend.onnx_graph_to_caffe2_net(onnx_model)
    caffe2_init_str = caffe2_init.SerializeToString()
    with open('weights/model.init.pb', "wb") as f:
        f.write(caffe2_init_str)
    caffe2_predict_str = caffe2_predict.SerializeToString()
    with open('weights/model.pred.pb', "wb") as f:
        f.write(caffe2_predict_str)

    # print benchmark
    model = model_helper.ModelHelper(name="model", init_params=False)

    init_net_proto = caffe2_pb2.NetDef()
    with open('weights/model.init.pb', "rb") as f:
        init_net_proto.ParseFromString(f.read())
    model.param_init_net = core.Net(init_net_proto)

    predict_net_proto = caffe2_pb2.NetDef()
    with open('weights/model.pred.pb', "rb") as f:
        predict_net_proto.ParseFromString(f.read())
    model.net = core.Net(predict_net_proto)

    model.param_init_net.GaussianFill([],
                                      model.net.external_inputs[0].GetUnscopedName(),
                                      shape=shape, mean=0.0, std=1.0)
    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net)
    workspace.BenchmarkNet(model.net.Proto().name, 5, 100, True)
    # remove temp data
    os.remove('weights/model.onnx')
    os.remove('weights/model.init.pb')
    os.remove('weights/model.pred.pb')


def weight_decay(model, decay=1e-5):
    p1 = []
    p2 = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            p1.append(param)
        else:
            p2.append(param)
    return [{'params': p1, 'weight_decay': 0.}, {'params': p2, 'weight_decay': decay}]


def accuracy(output, target, top_k):
    with torch.no_grad():
        max_k = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def resample():
    return random.choice((Image.BILINEAR, Image.BICUBIC))


def rotate(image, magnitude):
    magnitude = (magnitude / max_value) * 90

    if random.random() > 0.5:
        magnitude *= -1

    return image.rotate(magnitude, resample=resample())


def shear_x(image, magnitude):
    magnitude = (magnitude / max_value) * 0.3

    if random.random() > 0.5:
        magnitude *= -1

    return image.transform(image.size, Image.AFFINE, (1, magnitude, 0, 0, 1, 0), resample=resample())


def shear_y(image, magnitude):
    magnitude = (magnitude / max_value) * 0.3

    if random.random() > 0.5:
        magnitude *= -1

    return image.transform(image.size, Image.AFFINE, (1, 0, 0, magnitude, 1, 0), resample=resample())


def translate_x(image, magnitude):
    magnitude = (magnitude / max_value) * 0.5

    if random.random() > 0.5:
        magnitude *= -1

    pixels = magnitude * image.size[0]
    return image.transform(image.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), resample=resample())


def translate_y(image, magnitude):
    magnitude = (magnitude / max_value) * 0.5

    if random.random() > 0.5:
        magnitude *= -1

    pixels = magnitude * image.size[1]
    return image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), resample=resample())


def equalize(image, _):
    return ImageOps.equalize(image)


def invert(image, _):
    return ImageOps.invert(image)


def identity(image, _):
    return image


def normalize(image, _):
    return ImageOps.autocontrast(image)


def brightness(image, magnitude):
    if random.random() > 0.5:
        magnitude = (magnitude / max_value) * 1.8 + 0.1
        return ImageEnhance.Brightness(image).enhance(magnitude)
    else:
        magnitude = (magnitude / max_value) * 0.9

        if random.random() > 0.5:
            magnitude *= -1

        return ImageEnhance.Brightness(image).enhance(magnitude)


def color(image, magnitude):
    if random.random() > 0.5:
        magnitude = (magnitude / max_value) * 1.8 + 0.1
        return ImageEnhance.Color(image).enhance(magnitude)
    else:
        magnitude = (magnitude / max_value) * 0.9

        if random.random() > 0.5:
            magnitude *= -1

        return ImageEnhance.Color(image).enhance(magnitude)


def contrast(image, magnitude):
    if random.random() > 0.5:
        magnitude = (magnitude / max_value) * 1.8 + 0.1
        return ImageEnhance.Contrast(image).enhance(magnitude)
    else:
        magnitude = (magnitude / max_value) * 0.9

        if random.random() > 0.5:
            magnitude *= -1

        return ImageEnhance.Contrast(image).enhance(magnitude)


def sharpness(image, magnitude):
    if random.random() > 0.5:
        magnitude = (magnitude / max_value) * 1.8 + 0.1
        return ImageEnhance.Sharpness(image).enhance(magnitude)
    else:
        magnitude = (magnitude / max_value) * 0.9

        if random.random() > 0.5:
            magnitude *= -1

        return ImageEnhance.Sharpness(image).enhance(magnitude)


def solar(image, magnitude):
    magnitude = int((magnitude / max_value) * 256)
    if random.random() > 0.5:
        return ImageOps.solarize(image, magnitude)
    else:
        return ImageOps.solarize(image, 256 - magnitude)


def poster(image, magnitude):
    magnitude = int((magnitude / max_value) * 4)
    if random.random() > 0.5:
        if magnitude >= 8:
            return image
        return ImageOps.posterize(image, magnitude)
    else:
        if random.random() > 0.5:
            magnitude = 4 - magnitude
        else:
            magnitude = 4 + magnitude

        if magnitude >= 8:
            return image
        return ImageOps.posterize(image, magnitude)


def random_hsv(image):
    x = numpy.arange(0, 256, dtype=numpy.int16)
    hsv = numpy.random.uniform(-1, 1, 3) * [.015, .7, .4] + 1
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))

    lut_hue = ((x * hsv[0]) % 180).astype('uint8')
    lut_sat = numpy.clip(x * hsv[1], 0, 255).astype('uint8')
    lut_val = numpy.clip(x * hsv[2], 0, 255).astype('uint8')

    h = cv2.LUT(h, lut_hue)
    s = cv2.LUT(s, lut_sat)
    v = cv2.LUT(v, lut_val)

    image_hsv = cv2.merge((h, s, v)).astype('uint8')
    cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB, dst=image)


def random_affine(image):
    h = image.shape[0]
    w = image.shape[1]

    # Center
    center = numpy.eye(3)
    center[0, 2] = -image.shape[1] / 2  # x translation (pixels)
    center[1, 2] = -image.shape[0] / 2  # y translation (pixels)

    # Perspective
    perspective = numpy.eye(3)

    # Rotation and Scale
    rotation = numpy.eye(3)
    a = random.uniform(-30, 30)
    s = random.uniform(1 - 0.25, 1 + 0.25)
    rotation[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    shear = numpy.eye(3)
    shear[0, 1] = math.tan(random.uniform(-0.1, 0.1) * math.pi / 180)  # x shear (deg)
    shear[1, 0] = math.tan(random.uniform(-0.1, 0.1) * math.pi / 180)  # y shear (deg)

    # Translation
    translation = numpy.eye(3)
    translation[0, 2] = random.uniform(0.5 - 0.2, 0.5 + 0.2) * w  # x translation (pixels)
    translation[1, 2] = random.uniform(0.5 - 0.2, 0.5 + 0.2) * h  # y translation (pixels)

    # Combined rotation matrix, order of operations (right to left) is IMPORTANT
    matrix = translation @ shear @ rotation @ perspective @ center
    if (matrix != numpy.eye(3)).any():  # image changed
        image = cv2.warpAffine(image, matrix[:2], dsize=(w, h))  # affine
    return image


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        i, j, h, w = self.params(image.size)
        image = image.crop((j, i, j + w, i + h))
        return image.resize([self.size, self.size], resample())

    @staticmethod
    def params(size):
        scale = (0.08, 1.0)
        ratio = (3. / 4., 4. / 3.)
        for _ in range(10):
            target_area = random.uniform(*scale) * size[0] * size[1]
            aspect_ratio = math.exp(random.uniform(*(math.log(ratio[0]), math.log(ratio[1]))))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= size[0] and h <= size[1]:
                i = random.randint(0, size[1] - h)
                j = random.randint(0, size[0] - w)
                return i, j, h, w

        in_ratio = size[0] / size[1]
        if in_ratio < min(ratio):
            w = size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = size[1]
            w = int(round(h * max(ratio)))
        else:
            w = size[0]
            h = size[1]
        i = (size[1] - h) // 2
        j = (size[0] - w) // 2
        return i, j, h, w


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        self.num = self.num + n
        self.sum = self.sum + v * n
        self.avg = self.sum / self.num


class MixAugment:
    def __init__(self, mean=4, sigma=0.5, n=4):
        self.n = n
        self.mean = mean
        self.sigma = sigma
        self.transform = (equalize, identity, invert, normalize,
                          rotate, shear_x, shear_y, translate_x, translate_y,
                          brightness, color, contrast, sharpness, solar, poster)

    def __call__(self, image):
        aug_image = image.copy()

        for transform in numpy.random.choice(self.transform, self.n):
            magnitude = numpy.random.normal(self.mean, self.sigma)
            magnitude = min(max_value, max(0., magnitude))
            aug_image = transform(aug_image, magnitude)
        alpha = random.random()
        return Image.blend(image, aug_image, alpha if alpha > 0.3 else alpha / 3)


class RandomAffine:
    def __call__(self, image):
        image = numpy.asarray(image)
        random_hsv(image)
        image = random_affine(image)
        return Image.fromarray(image)


class RandomAugment:
    def __init__(self, mean=9, sigma=0.5, n=2):
        self.n = n
        self.mean = mean
        self.sigma = sigma
        self.transform = (equalize, identity, invert, normalize,
                          rotate, shear_x, shear_y, translate_x, translate_y,
                          brightness, color, contrast, sharpness, solar, poster)

    def __call__(self, image):
        for transform in numpy.random.choice(self.transform, self.n):
            magnitude = numpy.random.normal(self.mean, self.sigma)
            magnitude = min(max_value, max(0., magnitude))

            image = transform(image, magnitude)
        return image
