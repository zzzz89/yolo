# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import json
import math
import platform
import warnings
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp

from utils.datasets import exif_transpose, letterbox
from utils.general import (LOGGER, check_requirements, check_suffix, colorstr, increment_path, make_divisible,
                           non_max_suppression, scale_coords, xywh2xyxy, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import time_sync


def autopad(k, p=None):  # kernel, padding
    """
            用于Conv函数和Classify函数中,
            为same卷积或same池化作自动扩充（0填充）  Pad to 'same'
            根据卷积核大小k自动计算卷积核padding数（0填充）
            v3中只有两种卷积：
               1、下采样卷积:conv3x3 s=2 p=k//2=1
               2、feature size不变的卷积:conv1x1 s=1 p=k//2=1
            :params k: 卷积核的kernel_size
            :return p: 自动计算的需要pad值（0填充）
        """
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        """
               Standard convolution  conv+BN+act
               :params c1: 输入的channel值
               :params c2: 输出的channel值
               :params k: 卷积的kernel_size
               :params s: 卷积的stride
               :params p: 卷积的padding  一般是None  可以通过autopad自行计算需要pad的padding数
               :params g: 卷积的groups数  =1就是普通的卷积  >1就是深度可分离卷积,也就是分组卷积
               :params act: 激活函数类型   True就是SiLU()/Swish   False就是不使用激活函数
                            类型是nn.Module就使用传进来的激活函数类型
               """

        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)

        # Todo 修改激活函数
        # self.act = nn.Identity() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.Tanh() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.Sigmoid() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.LeakyReLU(0.1) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.Hardswish() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # 模型的前向传播
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
            用于Model类的fuse函数
            前向融合conv+bn计算 加速推理 一般用于测试/验证阶段
        """
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion

        """
               在yolo.py的parse_model中调用
               Standard bottleneck  Conv+Conv+shortcut

               :params c1: 第一个卷积的输入channel
               :params c2: 第二个卷积的输出channel
               :params shortcut: bool 是否有shortcut连接 默认是True
               :params g: 卷积分组的个数  =1就是普通卷积  >1就是深度可分离卷积
               :params e: expansion ratio  e*c2就是第一个卷积的输出channel=第二个卷积的输入channel
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class SPP(nn.Module):
    """
            空间金字塔池化 Spatial pyramid pooling layer used in YOLOv3-SPP
            :params c1: SPP模块的输入channel
            :params c2: SPP模块的输出channel
            :params k: 保存着三个maxpool的卷积核大小 默认是(5, 9, 13)
    """
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)   # 第一层卷积
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)   # 最后一层卷积  +1是因为有len(k)+1个输入
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    r""" 将宽高信息压缩到通道空间中。
    Focus 层通过将输入图像的四个象限拼接在一起，然后通过一个卷积层来提取特征。
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # 输入通道数, 输出通道数, 卷积核大小, 步幅, 填充, 分组数, 是否使用激活函数
        super().__init__()
        # 初始化卷积层，将四个象限拼接后的特征图映射到 c2 个通道
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # 注释掉了的 Contract 层用于将图像尺寸减半
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b, c, w, h) -> y(b, 4c, w/2, h/2)
        # 将输入特征图的四个象限拼接起来，并通过卷积层进行处理
        return self.conv(torch.cat([
            x[..., ::2, ::2],  # 左上象限
            x[..., 1::2, ::2],  # 右上象限
            x[..., ::2, 1::2],  # 左下象限
            x[..., 1::2, 1::2]   # 右下象限
        ], 1))
        # 使用 Contract 层（如果启用）将特征图尺寸减半
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    """
        用在yolo.py的parse_model模块
        改变输入特征的shape 将w和h维度(缩小)的数据收缩到channel维度上(放大)
        Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    """
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    """
        用在yolo.py的parse_model模块  用的不多
        Expand函数也是改变输入特征的shape，不过与Contract的相反， 是将channel维度(变小)的数据扩展到W和H维度(变大)。
        改变输入特征的shape 将channel维度(变小)的数据扩展到W和H维度(变大)
        Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    """

    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # 按照自身某个维度进行concat，常用来合并前后两个feature map，也就是上面Yolo 5s结构图中的Concat。
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class DetectMultiBackend(nn.Module):
    # MultiBackend 类用于在各种后端上进行 Python 推断
    def __init__(self, weights='yolov3.pt', device=None, dnn=True):
        # 用法:
        #   PyTorch:      weights = *.pt
        #   TorchScript:            *.torchscript.pt
        #   CoreML:                 *.mlmodel
        #   TensorFlow:             *_saved_model
        #   TensorFlow:             *.pb
        #   TensorFlow Lite:        *.tflite
        #   ONNX Runtime:           *.onnx
        #   OpenCV DNN:             *.onnx with dnn=True
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)  # 处理权重路径
        suffix, suffixes = Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '', '.mlmodel']
        check_suffix(w, suffixes)  # 检查权重后缀是否在允许的列表中
        pt, onnx, tflite, pb, saved_model, coreml = (suffix == x for x in suffixes)  # 后端布尔值
        jit = pt and 'torchscript' in w.lower()  # 判断是否为 TorchScript
        stride, names = 64, [f'class{i}' for i in range(1000)]  # 设置默认步幅和类别名称

        if jit:  # TorchScript
            LOGGER.info(f'Loading {w} for TorchScript inference...')  # 日志记录：加载 TorchScript 模型
            extra_files = {'config.txt': ''}  # 模型元数据
            model = torch.jit.load(w, _extra_files=extra_files)  # 加载 TorchScript 模型
            if extra_files['config.txt']:
                d = json.loads(extra_files['config.txt'])  # 解析额外的配置文件
                stride, names = int(d['stride']), d['names']  # 提取步幅和类别名称

        elif pt:  # PyTorch
            from models.experimental import attempt_load  # 导入 attempt_load 函数，避免循环导入
            model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)  # 加载 TorchScript 模型或 PyTorch 模型
            stride = int(model.stride.max())  # 获取模型的最大步幅
            names = model.module.names if hasattr(model, 'module') else model.names  # 获取类别名称，如果模型有 'module' 属性则从中获取类别名称
        elif coreml:  # CoreML *.mlmodel
            import coremltools as ct  # 导入 CoreML 工具包
            model = ct.models.MLModel(w)  # 加载 CoreML 模型
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f'Loading {w} for ONNX OpenCV DNN inference...')  # 日志：加载 ONNX 模型用于 OpenCV DNN 推理
            check_requirements(('opencv-python>=4.5.4',))  # 检查是否安装了 opencv-python 库
            net = cv2.dnn.readNetFromONNX(w)  # 使用 OpenCV DNN 读取 ONNX 模型
        elif onnx:  # ONNX Runtime
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')  # 日志：加载 ONNX 模型用于 ONNX Runtime 推理
            check_requirements(('onnx', 'onnxruntime-gpu' if torch.has_cuda else 'onnxruntime'))  # 检查是否安装了 onnx 和 onnxruntime 库
            import onnxruntime  # 导入 ONNX Runtime
            session = onnxruntime.InferenceSession(w, None)  # 创建 ONNX Runtime 推理会话
        else:  # TensorFlow 模型 (TFLite, pb, saved_model)
            # import tensorflow as tf  # 导入 TensorFlow 库
            # if pb:  # 如果是 TensorFlow Frozen Graph (.pb 文件)
            #     def wrap_frozen_graph(gd, inputs, outputs):
            #         # 包装冻结图
            #         x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # 包装函数
            #         # 剪枝操作，获取指定输入和输出
            #         return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
            #                        tf.nest.map_structure(x.graph.as_graph_element, outputs))
            #
            #     LOGGER.info(f'Loading {w} for TensorFlow *.pb inference...')  # 日志：加载 TensorFlow .pb 模型进行推理
            #     graph_def = tf.Graph().as_graph_def()  # 创建一个新的图形定义
            #     graph_def.ParseFromString(open(w, 'rb').read())  # 从 .pb 文件读取图形定义
            #     # 包装冻结图函数，指定输入和输出
            #     frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
            #
            # elif saved_model:  # 如果是 TensorFlow SavedModel
            #     LOGGER.info(f'Loading {w} for TensorFlow saved_model inference...')  # 日志：加载 TensorFlow SavedModel 进行推理
            #     model = tf.keras.models.load_model(w)  # 加载 SavedModel
            #
            # elif tflite:  # 如果是 TensorFlow Lite 模型
            #     if 'edgetpu' in w.lower():  # 如果是 Edge TPU 模型
            #         LOGGER.info(f'Loading {w} for TensorFlow Edge TPU inference...')  # 日志：加载 TensorFlow Edge TPU 模型进行推理
            #         import tflite_runtime.interpreter as tfli  # 导入 tflite_runtime
            #         # 根据平台选择相应的 Edge TPU 动态库
            #         delegate = {'Linux': 'libedgetpu.so.1',  # Linux 下的库
            #                     'Darwin': 'libedgetpu.1.dylib',  # macOS 下的库
            #                     'Windows': 'edgetpu.dll'}[platform.system()]  # Windows 下的库
            #         # 创建 Edge TPU 解释器
            #         interpreter = tfli.Interpreter(model_path=w, experimental_delegates=[tfli.load_delegate(delegate)])
            #     else:
            #         LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')  # 日志：加载 TensorFlow Lite 模型进行推理
            #         interpreter = tf.lite.Interpreter(model_path=w)  # 加载 TFLite 模型
            #     interpreter.allocate_tensors()  # 分配张量
            #     input_details = interpreter.get_input_details()  # 获取输入细节
            #     output_details = interpreter.get_output_details()  # 获取输出细节
            import tensorflow as tf
            if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
                def wrap_frozen_graph(gd, inputs, outputs):
                    x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                    return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                                   tf.nest.map_structure(x.graph.as_graph_element, outputs))

                LOGGER.info(f'Loading {w} for TensorFlow *.pb inference...')
                graph_def = tf.Graph().as_graph_def()
                graph_def.ParseFromString(open(w, 'rb').read())
                frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
            elif saved_model:
                LOGGER.info(f'Loading {w} for TensorFlow saved_model inference...')
                model = tf.keras.models.load_model(w)
            elif tflite:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
                if 'edgetpu' in w.lower():
                    LOGGER.info(f'Loading {w} for TensorFlow Edge TPU inference...')
                    import tflite_runtime.interpreter as tfli
                    delegate = {'Linux': 'libedgetpu.so.1',  # install https://coral.ai/software/#edgetpu-runtime
                                'Darwin': 'libedgetpu.1.dylib',
                                'Windows': 'edgetpu.dll'}[platform.system()]
                    interpreter = tfli.Interpreter(model_path=w, experimental_delegates=[tfli.load_delegate(delegate)])
                else:
                    LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')
                    interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
                interpreter.allocate_tensors()  # allocate
                input_details = interpreter.get_input_details()  # inputs
                output_details = interpreter.get_output_details()  # outputs
        self.__dict__.update(locals())  # 将所有局部变量赋值给实例属性

    def forward(self, im, augment=False, visualize=False, val=False):
        # MultiBackend 推理
        b, ch, h, w = im.shape  # 批量大小、通道数、高度、宽度

        if self.pt:  # PyTorch
            y = self.model(im) if self.jit else self.model(im, augment=augment, visualize=visualize)
            return y if val else y[0]

        elif self.coreml:  # CoreML *.mlmodel
            im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW 转为 numpy BHWC 格式 shape(1,320,192,3)
            im = Image.fromarray((im[0] * 255).astype('uint8'))  # 转换为 PIL 图像
            # im = im.resize((192, 320), Image.ANTIALIAS)  # （可选）调整图像大小
            y = self.model.predict({'image': im})  # 使用 CoreML 模型进行预测，返回的是 xywh 归一化坐标
            box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # 转换为 xyxy 像素坐标
            conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(float)  # 获取置信度和类别
            y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)  # 合并结果

        elif self.onnx:  # ONNX
            im = im.cpu().numpy()  # 将 torch 张量转为 numpy 数组
            if self.dnn:  # ONNX OpenCV DNN
                self.net.setInput(im)  # 设置输入
                y = self.net.forward()  # 前向推理
            else:  # ONNX Runtime
                y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im})[
                    0]  # 运行推理

        else:  # TensorFlow 模型（TFLite, pb, saved_model）
            im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW 转为 numpy BHWC 格式 shape(1,320,192,3)
            if self.pb:  # TensorFlow Frozen Graph
                y = self.frozen_func(x=self.tf.constant(im)).numpy()  # 执行推理
            elif self.saved_model:  # TensorFlow SavedModel
                y = self.model(im, training=False).numpy()  # 执行推理
            elif self.tflite:  # TensorFlow Lite
                input, output = self.input_details[0], self.output_details[0]
                int8 = input['dtype'] == np.uint8  # 是否是 TFLite 量化 uint8 模型
                if int8:
                    scale, zero_point = input['quantization']
                    im = (im / scale + zero_point).astype(np.uint8)  # 反量化
                self.interpreter.set_tensor(input['index'], im)  # 设置输入张量
                self.interpreter.invoke()  # 执行推理
                y = self.interpreter.get_tensor(output['index'])  # 获取输出张量
                if int8:
                    scale, zero_point = output['quantization']
                    y = (y.astype(np.float32) - zero_point) * scale  # 反量化
            y[..., 0] *= w  # 转换 x 方向坐标
            y[..., 1] *= h  # 转换 y 方向坐标
            y[..., 2] *= w  # 转换宽度
            y[..., 3] *= h  # 转换高度
        y = torch.tensor(y)  # 将结果转为 torch 张量
        return (y, []) if val else y  # 如果 val 为 True，则返回 (y, [])，否则只返回 y


class AutoShape(nn.Module):
    # 输入健壮的模型包装器，用于处理 cv2/np/PIL/torch 输入。包括预处理、推理和 NMS（非极大值抑制）
    conf = 0.25  # NMS 置信度阈值
    iou = 0.45  # NMS IoU 阈值
    classes = None  # （可选列表）按类别过滤，例如 COCO 中的人、猫和狗 = [0, 15, 16]
    multi_label = False  # NMS 允许每个框有多个标签
    max_det = 1000  # 每张图片的最大检测数量

    def __init__(self, model):
        super().__init__()
        self.model = model.eval()  # 设置模型为评估模式

    def autoshape(self):
        LOGGER.info('AutoShape already enabled, skipping... ')  # 模型已转换为 model.autoshape()，跳过
        return self

    def _apply(self, fn):
        # 对模型中非参数或未注册的缓冲区的张量应用 to()、cpu()、cuda()、half() 等方法
        self = super()._apply(fn)
        m = self.model.model[-1]  # Detect()
        m.stride = fn(m.stride)  # 应用函数到 stride
        m.grid = list(map(fn, m.grid))  # 应用函数到 grid 列表
        if isinstance(m.anchor_grid, list):
            m.anchor_grid = list(map(fn, m.anchor_grid))  # 应用函数到 anchor_grid 列表
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # 从各种输入源进行推断。对于 height=640，width=1280，RGB 图像的示例输入如下：
        #   file:       imgs = 'data/images/zidane.jpg'  # str 或 PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR 转 RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') 或 ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (缩放到 size=640，0-1 值)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # 图像列表

        t = [time_sync()]  # 记录开始时间
        p = next(self.model.parameters())  # 获取模型的设备和数据类型
        if isinstance(imgs, torch.Tensor):  # 如果输入是 torch.Tensor
            with amp.autocast(enabled=p.device.type != 'cpu'):  # 自动混合精度
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # 推断

        # 预处理
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # 图像数量和图像列表
        shape0, shape1, files = [], [], []  # 图像和推断形状，文件名
        for i, im in enumerate(imgs):
            f = f'image{i}'  # 文件名
            if isinstance(im, (str, Path)):  # 文件名或 URI
                im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                im = np.asarray(exif_transpose(im))
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # 图像在 CHW 格式
                im = im.transpose((1, 2, 0))  # 反转 dataloader .transpose(2, 0, 1)
            im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)  # 强制三通道输入
            s = im.shape[:2]  # HWC
            shape0.append(s)  # 图像形状
            g = (size / max(s))  # 缩放因子
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # 更新
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # 推断形状
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # 填充
        x = np.stack(x, 0) if n > 1 else x[0][None]  # 堆叠
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC 转 BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 转 fp16/32
        t.append(time_sync())  # 记录时间

        with amp.autocast(enabled=p.device.type != 'cpu'):  # 自动混合精度
            # 推断
            y = self.model(x, augment, profile)[0]  # 前向传播
            t.append(time_sync())  # 记录时间

            # 后处理
            y = non_max_suppression(y, self.conf, iou_thres=self.iou, classes=self.classes,
                                    multi_label=self.multi_label, max_det=self.max_det)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])  # 缩放坐标

            t.append(time_sync())  # 记录时间
            return Detections(imgs, y, files, t, self.names, x.shape)  # 返回检测结果


class Detections:
    r""" 用于推理结果的检测类。
    此类用于处理模型的推理输出，包括图像、预测框、文件名等信息，并提供归一化后的框坐标。
    """
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super().__init__()
        d = pred[0].device  # 获取设备类型（CPU或GPU）

        # 计算每张图像的归一化因子
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in imgs]  # 归一化因子

        # 初始化类属性
        self.imgs = imgs  # 图像列表，作为 numpy 数组
        self.pred = pred  # 预测结果列表，pred[0] 包含 (xyxy, conf, cls) 信息
        self.names = names  # 类别名称
        self.files = files  # 图像文件名列表
        self.xyxy = pred  # xyxy 像素坐标
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh 像素坐标
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy 归一化坐标
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh 归一化坐标
        self.n = len(self.pred)  # 图像数量（批次大小）
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # 时间戳（毫秒）
        self.s = shape  # 推理时的 BCHW 形状

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        r""" 显示、保存或裁剪检测结果。

        根据设置，执行以下操作：
        - `pprint`: 打印检测信息到日志。
        - `show`: 显示检测结果图像。
        - `save`: 保存检测结果图像到指定目录。
        - `crop`: 裁剪检测区域并保存。
        - `render`: 渲染检测结果到图像列表中。

        参数：
        - `save_dir`: 保存图像的目录路径。
        """
        crops = []  # 存储裁剪的检测区域
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            s = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # 生成图像信息字符串
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # 每个类别的检测数量
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # 将类别和数量添加到信息字符串
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))  # 初始化注释工具
                    for *box, conf, cls in reversed(pred):  # 逆序遍历预测框
                        label = f'{self.names[int(cls)]} {conf:.2f}'  # 标签
                        if crop:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({'box': box, 'conf': conf, 'cls': cls, 'label': label,
                                          'im': save_one_box(box, im, file=file, save=save)})
                        else:  # 其他操作
                            annotator.box_label(box, label, color=colors(cls))
                    im = annotator.im  # 更新图像
            else:
                s += '(no detections)'  # 没有检测到目标

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # 如果图像是 numpy 数组，则转换为 PIL 图像
            if pprint:
                LOGGER.info(s.rstrip(', '))  # 打印信息
            if show:
                im.show(self.files[i])  # 显示图像
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # 保存图像
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.imgs[i] = np.asarray(im)  # 渲染图像
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    def print(self):
        r""" 打印检测结果和处理速度信息。"""
        self.display(pprint=True)  # 打印检测结果
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' %
                    self.t)  # 打印每张图像的处理速度（预处理、推理、非极大值抑制）以及图像的形状
        # print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' %
        #             self.t)  # 打印每张图像的处理速度（预处理、推理、非极大值抑制）以及图像的形状

    def show(self):
        r""" 显示检测结果。"""
        self.display(show=True)  # 显示检测结果

    def save(self, save_dir='runs/detect/exp'):
        r""" 保存检测结果到指定目录。"""
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True)  # 递增目录名称
        self.display(save=True, save_dir=save_dir)  # 保存结果

    def crop(self, save=True, save_dir='runs/detect/exp'):
        r""" 裁剪检测结果并保存。"""
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True) if save else None
        return self.display(crop=True, save=save, save_dir=save_dir)  # 裁剪结果

    def render(self):
        r""" 渲染检测结果。"""
        self.display(render=True)  # 渲染结果
        return self.imgs  # 返回渲染后的图像

    def pandas(self):
        r""" 将检测结果转换为 pandas DataFrame 格式。"""
        new = copy(self)  # 返回对象的副本
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy 列名
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh 列名

        # 遍历 'xyxy', 'xyxyn', 'xywh', 'xywhn' 字段及其对应的列名
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            # 将检测结果转换为 DataFrame，并更新列名
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])  # 设置 DataFrame 属性

        return new  # 返回包含 DataFrame 的副本对象

    def tolist(self):
        r""" 返回一个 Detections 对象的列表。例如，可以用 'for result in results.tolist():' 遍历。"""
        # 创建一个 Detections 对象的列表，每个对象包含一个图像和对应的预测结果
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        # 对每个 Detections 对象，移除其内部列表，使其属性为单一元素
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # 从列表中弹出
        return x  # 返回包含 Detections 对象的列表

    def __len__(self):
        r""" 返回 Detections 对象中图像的数量。"""
        return self.n


class Classify(nn.Module):
    # 分类头，将输入 x(b,c1,20,20) 转换为 x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # 输入通道数, 输出通道数, 卷积核大小, 步幅, 填充, 分组
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化到 x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # 卷积层，将 x 转换为 x(b,c2,1,1)
        self.flat = nn.Flatten()  # 展平层

    def forward(self, x):
        # 如果 x 是列表，则对列表中的每个元素进行池化，并拼接成一个 tensor
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)
        return self.flat(self.conv(z))  # 卷积后展平为 x(b,c2)

