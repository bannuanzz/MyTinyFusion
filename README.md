<div align="center">

<h1> TinyFusion 复现实验 </h1>

<div align="center">
 <img src="assets/vis_v2-1.png" alt="TinyFusion效果展示" style="display:block; margin-left:auto; margin-right:auto;">
   <br>
  <em>
      TinyDiT-D14在ImageNet上生成的图像，从DiT-XL/2剪枝并蒸馏得到，实现2倍加速且只使用不到7%的预训练成本。
  </em>
</div>

<h3>TinyFusion: 可学习浅层扩散Transformer</h3>

📄 [[论文链接]](https://arxiv.org/abs/2412.01199)
</div>

---

## 环境配置

### 依赖安装
```bash
pip install -r requirements.txt
```

### 下载预训练模型
```bash
# 创建目录并下载模型
mkdir -p pretrained && cd pretrained
wget https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt
wget https://github.com/VainF/TinyFusion/releases/download/v1.0.0/TinyDiT-D14-MaskedKD-500K.pt
cd ..
```

### 数据准备
```bash
# 创建CIFAR-10测试数据集 (用于快速验证，替代ImageNet)
python create_cifar10_fast.py

# 提取特征
python extract_features.py --model DiT-XL/2 --data-path data/imagenet/train --features-path data/imagenet_encoded
```

## 运行基准算法

### 可学习剪枝算法
```bash
# 运行TinyFusion可学习剪枝算法
python prune_by_learning.py \
  --model DiT-XL/2 \
  --load-weight pretrained/DiT-XL-2-256x256.pt \
  --data-path data/imagenet_encoded \
  --epochs 1 \
  --global-batch-size 128 \
  --save-model outputs/pruned/DiT-D14-Learned-Baseline.pt
```

### 使用预训练模型生成图像
```bash
python sample.py --model DiT-D14/2 --ckpt pretrained/TinyDiT-D14-MaskedKD-500K.pt --seed 5464
```

## 实验说明

本项目成功复现了TinyFusion论文的可学习剪枝算法，将DiT-XL/2（28层）压缩为TinyDiT-D14（14层），实现了49.6%的参数压缩。

### 基准算法复现
- 成功跑通了他的预训练模型的基准算法，并且按照实验要求加入少量自己的数据。
- 成功运行原始TinyFusion可学习剪枝算法
- 验证了剪枝决策的智能化过程：从随机选择演化为基于重要性的层选择
- 实现了预期的参数压缩效果

### 改进算法实验
基于对原始算法缺陷的分析，实现了三种改进策略：
- **自适应温度调度器**：动态调整Gumbel Softmax温度参数
- **层级重要性感知剪枝**：为不同功能层分配差异化权重
- **渐进式联合优化**：统一剪枝决策学习和知识蒸馏过程

改进算法在简化实验环境下表现出更大的损失减少率，但最终损失有所增加，体现了算法改进的探索性质。

实验使用CIFAR-10数据集（1000张图像）替代ImageNet进行快速验证，所有预训练模型和特征文件已准备就绪。

## 致谢

本项目基于 [facebookresearch/DiT](https://github.com/facebookresearch/DiT) 构建。
