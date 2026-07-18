# YOLOv3 目标检测

基于 Ultralytics 风格的 **YOLOv3** 工程，支持 VOC 数据集训练、验证与推理（图片 / 视频 / 摄像头）。

仓库地址：https://github.com/zzzz89/yolo

---

## 仓库里有什么

本仓库**只包含代码与配置**，不包含大文件：

| 包含 | 不包含（需自行准备） |
|------|----------------------|
| `train.py` / `detect.py` / `val.py` | `weight/*.pt` 预训练权重 |
| `models/` 网络结构与模块 | `VOCdevkit/` 数据集 |
| `utils/` 工具库 | `runs/` 训练与检测输出 |
| `data/*.yaml` 数据与超参配置 | 自己训好的 `best.pt` |

---

## 目录结构

```
yolov3/
├── train.py                 # 训练
├── detect.py                # 推理（图/视频/摄像头）
├── val.py                   # 验证 mAP
├── Dataset_partitioning.py  # VOC XML → YOLO 标注并划分 train/val
├── requirements.txt
├── data/
│   ├── you.yaml             # 当前 VOC 数据集配置（20 类）
│   ├── voc.yaml / coco128.yaml
│   └── hyps/                # 超参数
├── models/
│   ├── yolov3.yaml          # 标准 YOLOv3（默认）
│   ├── yolov3-tiny.yaml
│   ├── yolov3-spp.yaml
│   ├── yolo.py / common.py / experimental.py
└── utils/                   # 数据加载、损失、指标、绘图等
```

本地还需自行准备（不在 Git 中）：

```
yolov3/weight/yolov3.pt              # 预训练权重
yolov3/VOCdevkit/images|labels/...   # 数据集
yolov3/runs/train/exp*/weights/best.pt  # 训练产物
```

---

## 环境

建议 Python 3.8+，有 NVIDIA GPU 更佳。

```bash
cd yolov3
pip install -r requirements.txt
```

需自行安装匹配的 **PyTorch + CUDA**（见 https://pytorch.org ）。

---

## 数据准备（VOC）

1. 将 VOC 放到 `yolov3/VOCdevkit/VOC2007/`（含 `Annotations`、`JPEGImages`）。
2. 在 `yolov3` 目录运行划分脚本：

```bash
python Dataset_partitioning.py
```

会生成 YOLO 格式标签，并划分到 `VOCdevkit/images/{train,val}`、`VOCdevkit/labels/{train,val}`。

3. 检查 `data/you.yaml` 中路径与类别是否正确（默认 20 类 VOC）。

---

## 训练

下载或放入预训练权重到 `weight/yolov3.pt`，然后：

```bash
cd yolov3
python train.py --weights weight/yolov3.pt --data data/you.yaml --cfg models/yolov3.yaml --epochs 50 --batch-size 8 --device 0
```

常用默认（已在 `train.py` 中配置过的思路）：

- `batch-size=8`：适合约 6GB 显存笔记本
- `cache=False`：避免把大量图片塞进内存导致卡顿
- `imgsz=416`

结果保存在：

```
runs/train/exp*/weights/best.pt   # 验证最好的权重（推荐用于推理）
runs/train/exp*/weights/last.pt   # 最后一轮
runs/train/exp*/results.png       # 曲线图
```

断点续训：

```bash
python train.py --resume
```

---

## 推理（检测）

### 图片

```bash
python detect.py --weights runs/train/exp6/weights/best.pt --source data/images
```

### 视频

```bash
python detect.py --weights runs/train/exp6/weights/best.pt --source your.mp4 --view-img
```

- `--view-img`：实时弹窗播放（不要写 `True/False`）
- `--conf-thres 0.25`：置信度阈值，越低框越多
- 结果在 `runs/detect/exp*/`

### 摄像头

```bash
python detect.py --weights runs/train/exp6/weights/best.pt --source 0 --view-img
```

> 将路径中的 `exp6` 换成你本机实际的训练目录名。若只有自行备份的 `best.pt`，直接把 `--weights` 指到该文件即可。

---

## 验证

```bash
python val.py --weights runs/train/exp*/weights/best.pt --data data/you.yaml
```

---

## 模型结构可选

| 配置 | 说明 |
|------|------|
| `models/yolov3.yaml` | 标准 YOLOv3（默认，精度优先） |
| `models/yolov3-tiny.yaml` | 更小更快 |
| `models/yolov3-spp.yaml` | 带 SPP 的变体 |

示例：

```bash
python train.py --cfg models/yolov3-tiny.yaml --weights weight/yolov3-tiny.pt
```

---

## 权重与数据集备份建议

Git 不适合存放大文件。建议：

1. **推理**：备份自己的 `best.pt`（约 118MB）到网盘 / U 盘  
2. **再训练**：另存 `VOCdevkit` + `weight/yolov3.pt`  
3. 换电脑时：clone 本仓库 + 拷回权重（和数据）即可

---

## 同步代码到 GitHub

在仓库根目录（含 `.git` 的那一层）执行：

```bash
git add .
git status
git commit -m "描述本次修改"
git push
```

Windows CMD 请先切到正确盘符目录，例如：

```cmd
cd /d d:\yolov123
```

---

## 说明

- 本项目基于 Ultralytics YOLOv3 风格实现，并针对较新的 NumPy / PyTorch / Pillow 做了兼容修改。
- 默认数据集配置为 VOC 20 类；换数据集时请改 `data/*.yaml` 中的路径、`nc` 与 `names`。
