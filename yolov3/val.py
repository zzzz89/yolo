# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
    这个文件主要是在每一轮训练结束后，验证当前模型的mAP、混淆矩阵等指标。

    实际上这个脚本最常用的应该是通过train.py调用 run 函数，而不是通过执行 val.py 的。

    所以在了解这个脚本的时候，其实最重要的就是 run 函数。

    难点：混淆矩阵+计算correct+计算mAP，一定要结合metrics.py脚本一起看
"""

import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
os.chdir(ROOT)  # 固定工作目录，任意位置启动都可

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.general import (LOGGER, NCOLS, box_iou, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync


def save_one_txt(predn, save_conf, shape, file):
    """
        函数功能：保存预测信息到txt文件
    """
    # 保存单个 txt 结果
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # 归一化增益，顺序为宽高宽高
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 归一化 xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # 标签格式
        with open(file, 'a') as f:  # 以追加模式打开文件
            f.write(('%g ' * len(line)).rstrip() % line + '\n')  # 将标签写入文件，每个值以空格分隔并以换行符结束


def save_one_json(predn, jdict, path, class_map):
    """
    代码的主要功能是将目标检测模型的预测结果格式化为 JSON 格式，以便于后续分析或存储。
    """
    # 保存单个 JSON 结果 {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem  # 获取图像 ID，如果文件名是数字，则转换为整数
    box = xyxy2xywh(predn[:, :4])  # 将预测的边界框从 xyxy 格式转换为 xywh 格式
    box[:, :2] -= box[:, 2:] / 2  # 将中心坐标转换为左上角坐标

    for p, b in zip(predn.tolist(), box.tolist()):  # 遍历每个预测结果和对应的边界框
        jdict.append({'image_id': image_id,  # 添加图像 ID
                      'category_id': class_map[int(p[5])],  # 获取类别 ID
                      'bbox': [round(x, 3) for x in b],  # 将边界框坐标四舍五入到小数点后三位
                      'score': round(p[4], 5)})  # 将置信度四舍五入到小数点后五位


def process_batch(detections, labels, iouv):
    """
    返回正确预测的矩阵。两个框集均采用 (x1, y1, x2, y2) 格式。
    参数：
        detections (Array[N, 6]): 预测框，格式为 x1, y1, x2, y2, 置信度, 类别
        labels (Array[M, 5]): 真实框，格式为 类别, x1, y1, x2, y2
    返回：
        correct (Array[N, 10]): 每个 IoU 阈值下的正确预测矩阵
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)  # 初始化正确预测矩阵
    iou = box_iou(labels[:, 1:], detections[:, :4])  # 计算真实框与预测框的 IoU
    # 找到 IoU 大于阈值并且类别匹配的预测
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))
    if x[0].shape[0]:  # 如果找到了匹配
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # 组合为 [标签, 检测, IoU]
        if x[0].shape[0] > 1:  # 如果有多个匹配
            matches = matches[matches[:, 2].argsort()[::-1]]  # 按照 IoU 从大到小排序
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # 保留唯一的检测框
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # 保留唯一的标签
        matches = torch.Tensor(matches).to(iouv.device)  # 转换回张量并移动到正确的设备
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv  # 更新正确预测矩阵
    return correct  # 返回正确预测矩阵


@torch.no_grad()
def run(data,
        weights=None,  # 模型权重路径 (model.pt)
        batch_size=32,  # 批处理大小
        imgsz=640,  # 推理时的图像尺寸 (像素)
        conf_thres=0.001,  # 置信度阈值
        iou_thres=0.6,  # NMS 的 IoU 阈值
        task='val',  # 任务类型：train, val, test, speed 或 study
        device='',  # 使用的设备，例如 0 或 0,1,2,3 或 cpu
        single_cls=False,  # 是否将数据集视为单类数据集
        augment=False,  # 是否进行增强推理
        verbose=False,  # 是否输出详细信息
        save_txt=False,  # 是否将结果保存到 *.txt 文件
        save_hybrid=False,  # 是否将标签+预测混合结果保存到 *.txt 文件
        save_conf=False,  # 是否在 --save-txt 中保存置信度
        save_json=False,  # 是否保存为 COCO-JSON 结果文件
        project=ROOT / 'runs/val',  # 保存到项目目录/project/name
        name='exp',  # 保存到项目目录/name
        exist_ok=False,  # 如果项目名称已存在，则不增量
        half=True,  # 是否使用 FP16 半精度推理
        dnn=False,  # 是否使用 OpenCV DNN 进行 ONNX 推理
        model=None,  # 指定模型
        dataloader=None,  # 指定数据加载器
        save_dir=Path(''),  # 保存目录
        plots=True,  # 是否生成可视化图表
        callbacks=Callbacks(),  # 回调函数
        compute_loss=None,  # 计算损失函数
        ):
    # 初始化/加载模型并设置设备
    training = model is not None  # 检查模型是否已初始化
    if training:  # 如果由 train.py 调用
        device, pt = next(model.parameters()).device, True  # 获取模型所在设备，并标记为 PyTorch 模型
        half &= device.type != 'cpu'  # 仅在 CUDA 上支持半精度
        model.half() if half else model.float()  # 根据是否使用半精度设置模型类型
    else:  # 如果直接调用
        device = select_device(device, batch_size=batch_size)  # 选择设备
        # 目录设置
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 增加运行目录
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 创建目录
        # 加载模型
        model = DetectMultiBackend(weights, device=device, dnn=dnn)  # 使用指定权重加载模型
        stride, pt = model.stride, model.pt  # 获取模型的步幅和是否为 PyTorch 模型
        imgsz = check_img_size(imgsz, s=stride)  # 检查图像尺寸是否有效
        half &= pt and device.type != 'cpu'  # 仅在 PyTorch 和非 CPU 设备上支持半精度
        if pt:
            model.model.half() if half else model.model.float()  # 设置模型为半精度或浮点精度
        else:
            half = False  # 非 PyTorch 后端不使用半精度
            batch_size = 1  # 导出模型默认批处理大小为 1
            device = torch.device('cpu')  # 强制使用 CPU 设备
            LOGGER.info(f'强制使用 --batch-size 1 正方形推理形状(1,3,{imgsz},{imgsz}) 对于非 PyTorch 后端')
        # 数据处理
        data = check_dataset(data)  # 检查数据集的有效性

    # 配置模型
    model.eval()  # 将模型设置为评估模式
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')  # 检查是否为 COCO 数据集
    nc = 1 if single_cls else int(data['nc'])  # 类别数量，如果是单类，则为 1，否则从数据中获取
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # 生成 IoU 向量，用于计算 mAP@0.5:0.95
    niou = iouv.numel()  # 获取 IoU 向量的元素数量

    # Dataloader
    if not training:  # 如果不是训练模式
        if pt and device.type != 'cpu':
            # 进行模型预热，使用全零张量以避免 CUDA 的延迟
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.model.parameters())))
        pad = 0.0 if task == 'speed' else 0.5  # 根据任务类型设置填充值
        task = task if task in ('train', 'val', 'test') else 'val'  # 确定任务类型，如果无效则默认为验证模式
        # 创建数据加载器
        dataloader = create_dataloader(data[task], imgsz, batch_size, stride, single_cls,
                                       pad=pad, rect=pt, prefix=colorstr(f'{task}: '))[0]
    seen = 0  # 初始化已处理的图像数量
    confusion_matrix = ConfusionMatrix(nc=nc)  # 初始化混淆矩阵，类别数量为 nc
    # 创建类别名称字典，如果模型有 names 属性则使用，否则使用 model.module.names
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    # 如果是 COCO 数据集，则将类别映射为 COCO 91 类，否则使用 1000 个类别
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    # 设置输出表头格式
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    # 初始化各项指标
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)  # 初始化损失张量，包含 3 个类别的损失
    jdict, stats, ap, ap_class = [], [], [], []  # 初始化 JSON 字典、统计信息和 AP 数据
    # 创建进度条
    pbar = tqdm(dataloader, desc=s, ncols=NCOLS, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # 进度条设置
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):  # 遍历数据加载器中的每个批次
        t1 = time_sync()  # 记录开始时间
        if pt:
            im = im.to(device, non_blocking=True)  # 将图像数据转移到指定设备
            targets = targets.to(device)  # 将目标数据转移到指定设备
        im = im.half() if half else im.float()  # 将图像转换为 FP16 或 FP32
        im /= 255  # 将像素值从 0-255 归一化到 0.0-1.0
        nb, _, height, width = im.shape  # 获取批次大小、通道数、高度和宽度
        t2 = time_sync()  # 记录处理时间
        dt[0] += t2 - t1  # 累加数据加载时间
        # 推理
        out, train_out = model(im) if training else model(im, augment=augment, val=True)  # 进行推理
        dt[1] += time_sync() - t2  # 累加推理时间
        # 损失计算
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # 计算损失并累加
        # NMS（非极大值抑制）
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # 将目标框转换为像素坐标
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # 为自动标注准备标签
        t3 = time_sync()  # 记录时间
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True,
                                  agnostic=single_cls)  # 进行非极大值抑制
        dt[2] += time_sync() - t3  # 累加 NMS 时间
        # 评估指标
        for si, pred in enumerate(out):  # 遍历每个预测结果
            labels = targets[targets[:, 0] == si, 1:]  # 获取当前图像的目标标签
            nl = len(labels)  # 标签数量
            tcls = labels[:, 0].tolist() if nl else []  # 目标类别
            path, shape = Path(paths[si]), shapes[si][0]  # 获取当前图像的路径和形状
            seen += 1  # 增加已处理的图像数量
            if len(pred) == 0:  # 如果没有检测到物体
                if nl:
                    stats.append(
                        (torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))  # 记录无预测情况
                continue  # 继续下一个图像
            # 处理预测结果
            if single_cls:
                pred[:, 5] = 0  # 将类别设为 0，表示单类检测
            predn = pred.clone()  # 克隆预测结果
            scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # 将预测框转换为原始图像坐标
            # 评估
            if nl:  # 如果有标签
                tbox = xywh2xyxy(labels[:, 1:5])  # 转换目标框格式
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # 转换为原始图像坐标
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # 合并标签类别和框
                correct = process_batch(predn, labelsn, iouv)  # 处理预测与标签的匹配
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)  # 更新混淆矩阵
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)  # 没有标签则初始化为全零
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # 记录评估结果
            # 保存/记录结果
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))  # 保存为文本文件
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # 追加到 COCO-JSON 字典
            callbacks.run('on_val_image_end', pred, predn, path, names, im[si])  # 运行回调
        # 绘制图像
        if plots and batch_i < 3:  # 仅绘制前 3 个批次
            f = save_dir / f'val_batch{batch_i}_labels.jpg'  # 标签图像保存路径
            Thread(target=plot_images, args=(im, targets, paths, f, names), daemon=True).start()  # 异步绘制标签图像
            f = save_dir / f'val_batch{batch_i}_pred.jpg'  # 预测图像保存路径
            Thread(target=plot_images, args=(im, output_to_target(out), paths, f, names),
                   daemon=True).start()  # 异步绘制预测图像

    # Compute metrics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # 将每个统计信息合并为 NumPy 数组
    if len(stats) and stats[0].any():  # 如果有统计信息且第一个元素有值
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)  # 计算每类的精度、召回率等
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5 和 AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()  # 平均精度、召回率
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # 每个类别的目标数量
    else:
        nt = torch.zeros(1)  # 如果没有目标，初始化为零

    # 打印结果
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # 打印格式
    # LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))  # 打印总体统计结果
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))  # 打印总体统计结果

    # 打印每个类别的结果
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            # LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))  # 打印每类的统计信息
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))  # 打印每类的统计信息

    # 打印处理速度
    t = tuple(x / seen * 1E3 for x in dt)  # 每张图像的处理速度（毫秒）
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)  # 输入图像的形状
        # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)  # 打印速度
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)  # 打印速度

    # 绘图
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))  # 绘制混淆矩阵
        callbacks.run('on_val_end')  # 运行结束时的回调

    # Save JSON
    if save_json and len(jdict):  # 如果要保存为 JSON 且结果字典不为空
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # 获取权重文件名
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # COCO 注释文件路径
        pred_json = str(save_dir / f"{w}_predictions.json")  # 保存预测结果的 JSON 文件路径
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')  # 日志输出信息
        with open(pred_json, 'w') as f:  # 打开文件以写入
            json.dump(jdict, f)  # 将结果字典写入 JSON 文件

        try:  # 尝试运行 COCO 评估
            check_requirements(['pycocotools'])  # 检查是否安装了 pycocotools
            from pycocotools.coco import COCO  # 导入 COCO API
            from pycocotools.cocoeval import COCOeval  # 导入 COCO 评估 API

            anno = COCO(anno_json)  # 初始化 COCO 注释 API
            pred = anno.loadRes(pred_json)  # 加载预测结果 API
            eval = COCOeval(anno, pred, 'bbox')  # 初始化 COCO 评估对象
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # 设置要评估的图像 ID
            eval.evaluate()  # 进行评估
            eval.accumulate()  # 计算统计数据
            eval.summarize()  # 输出评估结果摘要
            map, map50 = eval.stats[:2]  # 获取 mAP@0.5:0.95 和 mAP@0.5
        except Exception as e:  # 捕获异常
            LOGGER.info(f'pycocotools unable to run: {e}')  # 日志输出错误信息

    # 返回结果
    model.float()  # 将模型设置为浮点数模式，以便进行训练
    if not training:
        # 如果不是训练模式，输出保存的标签数量
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")  # 日志输出结果保存路径
    maps = np.zeros(nc) + map  # 初始化一个与类别数量相同的数组，并填充平均精度
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]  # 将每个类别的平均精度填入相应位置
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t  # 返回结果


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/you.yaml', help='dataset.yaml path')  # 数据集配置文件地址 包含数据集的路径、类别个数、类名、下载地址等信息
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp/weights/best.pt', help='model.pt path(s)')  #  模型的权重文件地址 weights
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')  # 前向传播的批次大小 默认32
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=416, help='inference size (pixels)')  #  输入网络的图片分辨率 默认640
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')  # object置信度阈值 默认0.25
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')  # 进行NMS时IOU的阈值 默认0.6
    parser.add_argument('--task', default='test', help='train, val, test, speed or study')  # 设置测试的类型 有train, val, test, speed or study几种 默认val
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')  # 测试的设备
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')  # 数据集是否只用一个类别 默认False
    parser.add_argument('--augment', action='store_true', help='augmented inference')  # 是否使用数据增强进行推理，默认为False
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')  # 是否打印出每个类别的mAP 默认False
    parser.add_argument('--save-txt', action='store_false', help='save results to *.txt')  #  是否以txt文件的形式保存模型预测框的坐标 默认False
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')  # 是否save label+prediction hybrid results to *.txt  默认False 是否将gt_label+pre_label一起输入nms
    parser.add_argument('--save-conf', action='store_false', help='save confidences in --save-txt labels')   # save-conf: 是否保存预测每个目标的置信度到预测txt文件中 默认False
    parser.add_argument('--save-json', action='store_false', help='save a COCO-JSON results file')   # 是否按照coco的json格式保存预测框，并且使用cocoapi做评估（需要同样coco的json格式的标签） 默认False
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')  # 测试保存的源文件 默认runs/val
    parser.add_argument('--name', default='exp', help='save to project/name')# name: 当前测试结果放在runs/val下的文件名  默认是exp
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')  # -exist-ok: 是否覆盖已有结果，默认为 False
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')  # half: 是否使用半精度 Float16 推理 可以缩短推理时间 但是默认是False
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')  # -dnn:是否使用 OpenCV DNN 进行 ONNX 推理，默认为 False
    opt = parser.parse_args()  # 解析上述参数
    opt.data = check_yaml(opt.data)  # 解析并检查参数文件（通常是 YAML 格式）
    opt.save_json |= opt.data.endswith('coco.yaml')  # 如果 opt.data 以 'coco.yaml' 结尾，则设置 save_json 为 True
    opt.save_txt |= opt.save_hybrid  # 如果 save_hybrid 为 True，则设置 save_txt 为 True
    print_args(FILE.stem, opt)  # 打印参数信息
    return opt


def main(opt):
    # 检测requirements文件中需要的包是否安装好了
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    # 如果task in ['train', 'val', 'test']就正常测试 训练集/验证集/测试集
    if opt.task in ('train', 'val', 'test'):  # 如果任务是 'train', 'val' 或 'test'，则正常运行
        if opt.conf_thres > 0.001:  # 如果置信度阈值大于 0.001（参见 https://github.com/ultralytics/yolov5/issues/1466）
            LOGGER.info(
                f'WARNING: confidence threshold {opt.conf_thres} >> 0.001 will produce invalid mAP values.')  # 记录警告信息，置信度阈值大于 0.001 会产生无效的 mAP 值
        run(**vars(opt))  # 运行程序，并将 opt 的属性作为参数传递

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]  # 确保权重参数是列表类型
        opt.half = True  # 启用半精度（FP16）以获得最快的结果
        if opt.task == 'speed':  # 如果任务是 'speed'，进行速度基准测试
            # 例如：python val.py --task speed --data coco.yaml --batch 1 --weights yolov3.pt yolov3-spp.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False  # 设置置信度阈值、IOU 阈值，不保存 JSON
            for opt.weights in weights:  # 遍历每个权重文件
                run(**vars(opt), plots=False)  # 运行程序，不生成图表
        elif opt.task == 'study':  # 如果任务是 'study'，进行速度与 mAP 的基准测试
            # 例如：python val.py --task study --data coco.yaml --iou 0.7 --weights yolov3.pt yolov3-spp.pt...
            for opt.weights in weights:  # 遍历每个权重文件
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # 生成保存结果的文件名
                x, y = list(range(256, 1536 + 128, 128)), []  # x 轴（图像尺寸），y 轴
                for opt.imgsz in x:  # 遍历每个图像尺寸
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')  # 记录当前运行的图像尺寸
                    r, _, t = run(**vars(opt), plots=False)  # 运行程序，不生成图表
                    y.append(r + t)  # 将结果和时间添加到 y 轴
                np.savetxt(f, y, fmt='%10.4g')  # 将结果保存到文件中
            os.system('zip -r study.zip study_*.txt')  # 将所有结果文件打包成 zip 文件
            plot_val_study(x=x)  # 绘制基准测试图表

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
