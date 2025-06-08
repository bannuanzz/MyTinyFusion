"""
改进版TinyFusion训练脚本
集成自适应温度调度和层级重要性感知剪枝
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
sys.path.append('/Users/morri/Desktop/Study/多媒体技术/TinyFusion')

from improvements.adaptive_temperature import AdaptiveTemperatureScheduler  
from improvements.layer_importance import LayerImportanceAwarePruning
from models_with_layer_pruning import DiTWithLayerPruning
from train_masked_kd import create_model, load_dataset

class ImprovedTinyFusionTrainer:
    """
    改进版TinyFusion训练器
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化改进组件
        self.temp_scheduler = AdaptiveTemperatureScheduler(
            initial_tau=5.0,
            min_tau=0.1, 
            confidence_threshold=0.8
        )
        
        self.importance_pruning = LayerImportanceAwarePruning(
            total_layers=28
        )
        
        # 初始化模型
        self.teacher_model = create_model(config, is_teacher=True)
        self.student_model = create_model(config, is_teacher=False)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 损失记录
        self.train_losses = []
        self.temperature_history = []
        self.confidence_history = []
        
    def compute_improved_loss(self, student_outputs, teacher_outputs, pruning_gates):
        """
        计算改进版损失函数
        """
        # 基础知识蒸馏损失
        kd_loss = nn.MSELoss()(student_outputs, teacher_outputs.detach())
        
        # 剪枝损失（Gumbel Softmax）
        pruning_loss = 0.0
        for gate in pruning_gates:
            # 计算熵损失，鼓励决策明确性
            probs = torch.softmax(gate, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            pruning_loss += torch.mean(entropy)
        
        # 层级重要性感知损失
        importance_weighted_loss = self.importance_pruning.get_weighted_pruning_loss(
            pruning_gates, kd_loss
        )
        
        total_loss = importance_weighted_loss + 0.1 * pruning_loss
        return total_loss, kd_loss, pruning_loss
        
    def train_epoch(self, dataloader, epoch, total_epochs):
        """
        训练一个epoch
        """
        self.student_model.train()
        epoch_loss = 0.0
        epoch_kd_loss = 0.0
        epoch_pruning_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            with torch.no_grad():
                teacher_outputs = self.teacher_model(data)
                
            student_outputs, pruning_gates = self.student_model.forward_with_gates(data)
            
            # 更新温度
            current_tau = self.temp_scheduler.update_temperature(
                torch.cat([gate.flatten() for gate in pruning_gates]), 
                epoch, 
                total_epochs
            )
            
            # 更新模型中的温度参数
            self.student_model.set_gumbel_temperature(current_tau)
            
            # 计算改进版损失
            total_loss, kd_loss, pruning_loss = self.compute_improved_loss(
                student_outputs, teacher_outputs, pruning_gates
            )
            
            # 反向传播
            total_loss.backward()
            self.optimizer.step()
            
            # 记录损失
            epoch_loss += total_loss.item()
            epoch_kd_loss += kd_loss.item()
            epoch_pruning_loss += pruning_loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}: '
                      f'Loss={total_loss.item():.4f}, '
                      f'Temp={current_tau:.4f}, '
                      f'Confidence={self.temp_scheduler.confidence_history[-1]:.4f}')
        
        # 记录epoch统计信息
        avg_loss = epoch_loss / len(dataloader)
        avg_kd_loss = epoch_kd_loss / len(dataloader)
        avg_pruning_loss = epoch_pruning_loss / len(dataloader)
        
        self.train_losses.append(avg_loss)
        self.temperature_history.append(current_tau)
        
        return avg_loss, avg_kd_loss, avg_pruning_loss
        
    def evaluate(self, dataloader):
        """
        评估模型性能
        """
        self.student_model.eval()
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                teacher_outputs = self.teacher_model(data)
                student_outputs, pruning_gates = self.student_model.forward_with_gates(data)
                
                loss, _, _ = self.compute_improved_loss(
                    student_outputs, teacher_outputs, pruning_gates
                )
                
                total_loss += loss.item() * data.size(0)
                num_samples += data.size(0)
        
        return total_loss / num_samples
        
    def get_pruning_statistics(self):
        """
        获取剪枝统计信息
        """
        self.student_model.eval()
        pruning_decisions = []
        
        with torch.no_grad():
            # 使用一个dummy input来获取剪枝决策
            dummy_input = torch.randn(1, 3, 256, 256).to(self.device)
            _, gates = self.student_model.forward_with_gates(dummy_input)
            
            for gate in gates:
                probs = torch.softmax(gate, dim=-1)
                decisions = torch.argmax(probs, dim=-1)
                pruning_decisions.append(decisions.cpu().numpy())
        
        return pruning_decisions
        
    def save_model(self, path):
        """
        保存模型
        """
        torch.save({
            'student_model': self.student_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'temperature_scheduler': {
                'tau': self.temp_scheduler.tau,
                'confidence_history': self.temp_scheduler.confidence_history
            },
            'train_losses': self.train_losses,
            'temperature_history': self.temperature_history
        }, path)
        
    def load_model(self, path):
        """
        加载模型
        """
        checkpoint = torch.load(path)
        self.student_model.load_state_dict(checkpoint['student_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.temp_scheduler.tau = checkpoint['temperature_scheduler']['tau']
        self.temp_scheduler.confidence_history = checkpoint['temperature_scheduler']['confidence_history']
        self.train_losses = checkpoint['train_losses']
        self.temperature_history = checkpoint['temperature_history']

def main():
    """
    主训练函数
    """
    # 配置参数（适配2-GPU环境）
    class Config:
        data_path = "/Users/morri/Desktop/Study/多媒体技术/TinyFusion/cifar10_test_dataset"
        image_size = 256
        num_classes = 10
        batch_size = 32  # 原始128的1/4
        learning_rate = 1e-4
        weight_decay = 1e-4
        num_epochs = 50
        device = "cuda"
        model_name = "DiT-XL/2"
        lora_rank = 8  # 原始16的1/2
        
    config = Config()
    
    # 加载数据
    train_loader = load_dataset(config, split='train')
    val_loader = load_dataset(config, split='val')
    
    # 初始化训练器
    trainer = ImprovedTinyFusionTrainer(config)
    
    print("开始改进版TinyFusion训练...")
    print(f"使用设备: {trainer.device}")
    print(f"批大小: {config.batch_size}")
    print(f"训练轮数: {config.num_epochs}")
    
    # 训练循环
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        print(f"\n=== Epoch {epoch+1}/{config.num_epochs} ===")
        
        # 训练
        train_loss, kd_loss, pruning_loss = trainer.train_epoch(
            train_loader, epoch, config.num_epochs
        )
        
        # 验证
        val_loss = trainer.evaluate(val_loader)
        
        print(f"训练损失: {train_loss:.4f}")
        print(f"  - KD损失: {kd_loss:.4f}")
        print(f"  - 剪枝损失: {pruning_loss:.4f}")
        print(f"验证损失: {val_loss:.4f}")
        print(f"当前温度: {trainer.temp_scheduler.get_temperature():.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_model('improved_tinyfusion_best.pth')
            print(f"保存最佳模型 (验证损失: {val_loss:.4f})")
        
        # 每10轮打印剪枝统计
        if (epoch + 1) % 10 == 0:
            pruning_stats = trainer.get_pruning_statistics()
            print(f"剪枝统计: {pruning_stats}")
    
    print("\n训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    
    # 保存最终模型
    trainer.save_model('improved_tinyfusion_final.pth')
    
    # 生成训练报告
    generate_training_report(trainer, config)

def generate_training_report(trainer, config):
    """
    生成训练报告
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 损失曲线
    axes[0, 0].plot(trainer.train_losses)
    axes[0, 0].set_title('训练损失曲线')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # 温度变化曲线
    axes[0, 1].plot(trainer.temperature_history)
    axes[0, 1].set_title('温度调度曲线')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Temperature')
    axes[0, 1].grid(True)
    
    # 置信度历史
    if trainer.temp_scheduler.confidence_history:
        axes[1, 0].plot(trainer.temp_scheduler.confidence_history)
        axes[1, 0].set_title('决策置信度历史') 
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Confidence')
        axes[1, 0].grid(True)
    
    # 剪枝统计
    pruning_stats = trainer.get_pruning_statistics()
    if pruning_stats:
        layer_indices = range(len(pruning_stats))
        pruning_decisions = [np.mean(stat) for stat in pruning_stats]
        axes[1, 1].bar(layer_indices, pruning_decisions)
        axes[1, 1].set_title('层级剪枝决策分布')
        axes[1, 1].set_xlabel('Layer Index')
        axes[1, 1].set_ylabel('Average Pruning Decision')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('improved_tinyfusion_training_report.png', dpi=300)
    print("训练报告图表已保存为: improved_tinyfusion_training_report.png")

if __name__ == "__main__":
    main()
