# Mask R-CNN 与 Sparse R-CNN 在 VOC 数据集上的训练与测试
## 项目概述
本实验使用 mmdetection 框架在 PASCAL VOC 数据集上训练并对比了 Mask R-CNN 和 Sparse R-CNN 模型，完成了以下任务：
- 在 VOC 数据集上训练并测试两种模型
- 可视化对比两种模型的 proposal boxes 和预测结果
- 在 VOC 外的新图像上测试模型泛化能力
## 依赖安装
```bash
# 创建并激活虚拟环境
conda create -n mmdet python=3.8 -y
conda activate mmdet

# 安装 PyTorch
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# 安装 mmcv
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

# 克隆 mmdetection 仓库
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection

# 安装依赖
pip install -r requirements/build.txt
pip install -v -e .
```
## 数据集准备
- 下载地址
  ```bash
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar 
  ```
- 创建数据集目录结构
  ```bash
  mmdetection
  ├── data
  │   └── VOCdevkit
  │       ├── VOC2007
  │       └── VOC2012
  ```
## 模型训练
- MASK
  ```bash
  python tools/train.py configs/mask_rcnn/mask_rcnn_r50_fpn_1x_voc.py 
  ```
- SPARSE
  ```bash
  python tools/train.py configs/sparsek_rcnn/sparse_rcnn_r50_fpn_1x_voc.py 
  ```
  # 训练参数配置

- 参数配置：
 ```text
| 参数         | 值         |
|--------------|------------|
| Batch size   | 8 (每 GPU) |
| 优化器       | SGD        |
| 学习率       | 0.02       |
| 动量         | 0.9        |
| 权重衰减     | 0.0001     |
| Epochs       | 12         |
| 损失函数     | 分类交叉熵 + 回归 L1 损失 |
```
## 测试集评估
```bash
# Mask R-CNN
.python tools/test.py configs/mask_rcnn/mask_rcnn_r50_fpn_1x_voc.py work_dirs/mask_rcnn/latest.pth 2 --eval bbox segm

# Sparse R-CNN
python tools/test.py configs/sparse_rcnn/sparse_rcnn_r50_fpn_1x_voc.py work_dirs/sparse_rcnn/latest.pth 2 --eval bbox segm
```
## 结果可视化
```bash
# Mask R-CNN 可视化
python tools/analysis_tools/analyze_results.py \
    configs/mask_rcnn/mask_rcnn_r50_fpn_1x_voc.py \
    work_dirs/mask_rcnn/latest.pth \
    --show-dir vis_results/mask_rcnn

# Sparse R-CNN 可视化
python tools/analysis_tools/analyze_results.py \
    configs/sparse_rcnn/sparse_rcnn_r50_fpn_1x_voc.py \
    work_dirs/sparse_rcnn/latest.pth \
    --show-dir vis_results/sparse_rcnn
```
## 自定义图像
```bash
# 单张图像推理
python demo/image_demo.py \
    ${IMAGE_PATH} \
    configs/mask_rcnn/mask_rcnn_r50_fpn_1x_voc.py \
    work_dirs/mask_rcnn/latest.pth \
    --out-file ${OUTPUT_PATH}
```
## 项目结构
```text
mmdetection/
├── configs/              # 模型配置文件
├── data/                 # 数据集
├── demo/                 # 演示脚本
├── docs/                 # 文档
├── mmdet/                # 核心代码
├── tools/                # 训练/测试工具
├── work_dirs/            # 训练输出目录
│   ├── mask_rcnn/        # Mask R-CNN 训练结果
│   └── sparse_rcnn/      # Sparse R-CNN 训练结果
├── vis_results/          # 可视化结果
├── README.md             # 本文件
└── requirements.txt      # 依赖列表
```
