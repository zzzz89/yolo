# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
Model validation metrics
"""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def fitness(x):
    # 模型适应度，作为指标的加权组合
    w = [0.0, 0.0, 0.1, 0.9]  # 权重对应于 [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)  # 返回加权后的适应度


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
    """
    计算每个类别的平均精度，给定召回率和精度曲线。
    来源: https://github.com/rafaelpadilla/Object-Detection-Metrics。

    # 参数
        tp:  真阳性 (nparray, nx1 或 nx10)。
        conf:  物体置信度值，范围从 0 到 1 (nparray)。
        pred_cls:  预测的物体类别 (nparray)。
        target_cls:  真实物体类别 (nparray)。
        plot:  是否绘制精度-召回曲线在 mAP@0.5。
        save_dir:  绘图保存目录。

    # 返回
        根据 py-faster-rcnn 计算的平均精度。
    """

    # 按物体置信度排序
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # 找到唯一类别
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # 类别数量，检测数量

    # 创建精度-召回曲线并为每个类别计算 AP
    px, py = np.linspace(0, 1, 1000), []  # 用于绘图
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))

    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # 标签数量
        n_p = i.sum()  # 预测数量

        if n_p == 0 or n_l == 0:
            continue
        else:
            # 累积假阳性和真阳性
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # 召回率
            recall = tpc / (n_l + 1e-16)  # 召回率曲线
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # 负 x，xp 因为 xp 递减

            # 精度
            precision = tpc / (tpc + fpc)  # 精度曲线
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p 在 pr_score

            # 从召回-精度曲线计算 AP
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # mAP@0.5 的精度

    # 计算 F1 (精度和召回的调和平均)
    f1 = 2 * p * r / (p + r + 1e-16)
    names = [v for k, v in names.items() if k in unique_classes]  # 列表: 仅包含有数据的类别
    names = {i: v for i, v in enumerate(names)}  # 转为字典

    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # 最大 F1 索引
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ 计算平均精度，给定召回率和精度曲线
    # 参数
        recall:    召回率曲线（列表）
        precision: 精度曲线（列表）
    # 返回
        平均精度，精度曲线，召回率曲线
    """
    # 在开始和结束处添加哨兵值
    mrec = np.concatenate(([0.0], recall, [1.0]))  # 召回率的哨兵值
    mpre = np.concatenate(([1.0], precision, [0.0]))  # 精度的哨兵值

    # 计算精度包络线
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))  # 反转并计算最大累积值

    # 积分曲线下面积
    method = 'interp'  # 方法：'continuous'，'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101点插值（COCO）
        ap = np.trapezoid(np.interp(x, mrec, mpre), x)  # 积分（NumPy 2.x）
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # 召回率变化的点
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # 曲线下面积

    return ap, mpre, mrec  # 返回平均精度、精度曲线和召回率曲线


class ConfusionMatrix:
    # 更新版本，来源于 https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        """
        初始化混淆矩阵。

        # 参数
            nc: 类别数量。
            conf: 置信度阈值，低于该值的检测将被忽略。
            iou_thres: IoU 阈值，用于确定匹配的真阳性。
        """
        self.matrix = np.zeros((nc + 1, nc + 1))  # 初始化混淆矩阵
        self.nc = nc  # 类别数量
        self.conf = conf  # 置信度阈值
        self.iou_thres = iou_thres  # IoU 阈值

    def process_batch(self, detections, labels):
        """
        处理一批检测和标签数据，更新混淆矩阵。

        # 参数
            detections (Array[N, 6]): 形状为 (N, 6)，包含 [x1, y1, x2, y2, conf, class]。
            labels (Array[M, 5]): 形状为 (M, 5)，包含 [class, x1, y1, x2, y2]。

        # 返回
            None, 根据检测结果更新混淆矩阵。
        """
        # 过滤掉置信度低于阈值的检测
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()  # 真实类别
        detection_classes = detections[:, 5].int()  # 预测类别
        iou = box_iou(labels[:, 1:], detections[:, :4])  # 计算 IoU

        x = torch.where(iou > self.iou_thres)  # 找到 IoU 大于阈值的匹配
        if x[0].shape[0]:
            # 合并匹配索引和 IoU 值
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                # 按照 IoU 值排序并去重
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))  # 没有匹配的情况

        n = matches.shape[0] > 0  # 是否有匹配
        m0, m1, _ = matches.transpose().astype(np.int16)  # 分离匹配索引
        for i, gc in enumerate(gt_classes):  # 遍历真实类别
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # 正确的预测
            else:
                self.matrix[self.nc, gc] += 1  # 背景误报

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # 背景漏报

    def matrix(self):
        """ 返回混淆矩阵。 """
        return self.matrix

    def plot(self, normalize=True, save_dir='', names=()):
        """
        绘制混淆矩阵。

        # 参数
            normalize: 是否归一化混淆矩阵。
            save_dir: 保存目录。
            names: 类别名称，用于标签。
        """
        try:
            import seaborn as sn

            array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-6) if normalize else 1)  # 归一化列
            array[array < 0.005] = np.nan  # 小于阈值的元素不注释

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # 设置标签大小
            labels = (0 < len(names) < 99) and len(names) == self.nc  # 应用名称到坐标标签
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # 忽略空矩阵的警告
                sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                           xticklabels=names + ['background FP'] if labels else "auto",
                           yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)  # 保存混淆矩阵图
            plt.close()
        except Exception as e:
            print(f'WARNING: ConfusionMatrix plot failure: {e}')

    def print(self):
        """ 打印混淆矩阵。 """
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))



def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # 计算 box1 和 box2 的 IoU。box1 是 4 个元素，box2 是 nx4
    box2 = box2.T  # 转置 box2，以便于后续计算

    # 获取边界框的坐标
    if x1y1x2y2:  # 如果是 (x1, y1, x2, y2) 格式
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # 将 (xywh) 转换为 (xyxy) 格式
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # 计算交集面积
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # 计算并集面积
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps  # 加上 eps 避免除零

    iou = inter / union  # 计算 IoU
    if GIoU or DIoU or CIoU:  # 如果需要计算 GIoU、DIoU 或 CIoU
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # 最小外接框宽度
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # 最小外接框高度
        if CIoU or DIoU:  # 距离或完全 IoU
            c2 = cw ** 2 + ch ** 2 + eps  # 外接框对角线的平方
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                     (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # 中心距离的平方
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # CIoU 计算
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU 计算
            c_area = cw * ch + eps  # 外接框面积
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # 返回普通 IoU



def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    返回交并比（Jaccard 指数）值。
    预期两组框的格式为 (x1, y1, x2, y2)。
    参数：
        box1 (Tensor[N, 4]): 第一个框的张量
        box2 (Tensor[M, 4]): 第二个框的张量
    返回：
        iou (Tensor[N, M]): 包含 boxes1 和 boxes2 中每对元素的成对 IoU 值的 NxM 矩阵
    """
    def box_area(box):
        # 计算框的面积
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)  # 计算 box1 的面积
    area2 = box_area(box2.T)  # 计算 box2 的面积

    # 计算交集面积
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # 计算 IoU = inter / (area1 + area2 - inter)



def bbox_ioa(box1, box2, eps=1E-7):
    """
    返回 box1 与 box2 的交集占 box2 面积的比率。框的格式为 x1y1x2y2。
    参数：
        box1: np.array，形状为 (4)，表示单个框
        box2: np.array，形状为 (nx4)，表示多个框
    返回：
        np.array，形状为 (n)，表示每个 box2 的交集占其面积的比率
    """
    box2 = box2.transpose()  # 转置 box2，方便后续处理
    # 获取框的坐标
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]  # box1 的坐标
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]  # box2 的坐标
    # 计算交集面积
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)
    # 计算 box2 面积
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps  # 加上 eps 避免除零
    # 返回交集占 box2 面积的比率
    return inter_area / box2_area

def wh_iou(wh1, wh2):
    # 返回 nxm 的 IoU 矩阵。wh1 是 nx2 的宽高数组，wh2 是 mx2 的宽高数组。
    wh1 = wh1[:, None]  # 将 wh1 转换为 [N, 1, 2] 形状
    wh2 = wh2[None]  # 将 wh2 转换为 [1, M, 2] 形状
    inter = torch.min(wh1, wh2).prod(2)  # 计算交集面积，结果为 [N, M]
    # 计算 IoU = 交集面积 / (区域1面积 + 区域2面积 - 交集面积)
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # 返回 IoU 矩阵


# Plots ----------------------------------------------------------------------------------------------------------------
def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # 绘制精确度-召回率曲线
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)  # 创建子图
    py = np.stack(py, axis=1)  # 将 py 转换为二维数组

    if 0 < len(names) < 21:  # 如果类别数量小于 21，则显示每类的图例
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # 绘制每个类的 (召回, 精确度) 曲线
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # 绘制所有类的平均曲线，颜色为灰色

    # 绘制所有类的平均精确度
    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')  # 设置 x 轴标签
    ax.set_ylabel('Precision')  # 设置 y 轴标签
    ax.set_xlim(0, 1)  # 设置 x 轴范围
    ax.set_ylim(0, 1)  # 设置 y 轴范围
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")  # 添加图例
    fig.savefig(Path(save_dir), dpi=250)  # 保存图像
    plt.close()  # 关闭图像


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # 绘制度量-置信度曲线
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)  # 创建子图

    if 0 < len(names) < 21:  # 如果类别数量小于 21，则显示每类的图例
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # 绘制每个类的 (置信度, 度量) 曲线
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # 绘制所有类的曲线，颜色为灰色

    y = py.mean(0)  # 计算所有类的平均度量
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)  # 设置 x 轴标签
    ax.set_ylabel(ylabel)  # 设置 y 轴标签
    ax.set_xlim(0, 1)  # 设置 x 轴范围
    ax.set_ylim(0, 1)  # 设置 y 轴范围
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")  # 添加图例
    fig.savefig(Path(save_dir), dpi=250)  # 保存图像
    plt.close()  # 关闭图像