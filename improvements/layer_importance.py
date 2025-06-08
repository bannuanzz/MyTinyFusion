"""
层级重要性感知剪枝改进
解决剪枝时缺乏层重要性先验知识的问题
"""
import torch
import torch.nn as nn
import numpy as np

class LayerImportanceAwarePruning:
    """
    基于层级重要性的剪枝策略
    """
    def __init__(self, total_layers=28, architecture='DiT'):
        self.total_layers = total_layers
        self.architecture = architecture
        self.importance_weights = self._compute_layer_importance()
        self.pruning_history = []
        
    def _compute_layer_importance(self):
        """
        计算不同层的重要性权重
        基于DiT架构的经验和理论分析
        """
        weights = np.ones(self.total_layers)
        
        if self.architecture == 'DiT':
            # DiT架构的层级重要性分析
            
            # 早期层（0-6层）：空间特征提取 - 较高重要性
            early_layers = min(7, self.total_layers)
            weights[:early_layers] *= 1.3
            
            # 中间层（7-20层）：特征变换和交互 - 基准重要性
            mid_start = early_layers
            mid_end = min(21, self.total_layers)
            weights[mid_start:mid_end] *= 1.0
            
            # 后期层（21-27层）：高级语义理解 - 最高重要性
            if self.total_layers > 21:
                weights[21:] *= 1.5
                
        # 考虑层间依赖性：相邻层的重要性应该相似
        smoothed_weights = self._smooth_importance_weights(weights)
        
        # 归一化
        smoothed_weights = smoothed_weights / np.mean(smoothed_weights)
        return torch.tensor(smoothed_weights, dtype=torch.float32)
    
    def _smooth_importance_weights(self, weights, alpha=0.1):
        """
        平滑重要性权重，考虑层间依赖性
        """
        smoothed = weights.copy()
        for i in range(1, len(weights) - 1):
            neighbor_avg = (weights[i-1] + weights[i+1]) / 2
            smoothed[i] = (1 - alpha) * weights[i] + alpha * neighbor_avg
        return smoothed
    
    def get_weighted_pruning_loss(self, pruning_decisions, base_loss):
        """
        计算考虑层重要性的剪枝损失
        """
        importance_penalty = 0.0
        layer_idx = 0
        
        for decision in pruning_decisions:
            if isinstance(decision, torch.Tensor) and decision.numel() > 0:
                # 获取决策的形状
                if len(decision.shape) == 1:
                    # 单个决策
                    num_choices = decision.shape[0]
                    batch_size = 1
                elif len(decision.shape) == 2:
                    # 批次决策
                    batch_size, num_choices = decision.shape
                else:
                    continue
                
                # 确保不超出层数范围
                end_idx = min(layer_idx + num_choices, self.total_layers)
                if layer_idx >= end_idx:
                    continue
                    
                # 获取对应层的重要性权重
                layer_importance = self.importance_weights[layer_idx:end_idx]
                
                # 计算剪枝概率（0表示保留，1表示剪枝）
                probs = torch.softmax(decision, dim=-1)
                if len(probs.shape) == 1:
                    probs = probs.unsqueeze(0)
                
                # 计算每层被剪枝的概率
                pruning_probs = 1.0 - probs[:, :len(layer_importance)]
                
                # 计算重要性加权惩罚
                if layer_importance.device != pruning_probs.device:
                    layer_importance = layer_importance.to(pruning_probs.device)
                
                weighted_penalty = torch.sum(
                    pruning_probs * layer_importance.unsqueeze(0), 
                    dim=-1
                )
                importance_penalty += torch.mean(weighted_penalty)
                
                layer_idx = end_idx
        
        # 记录剪枝历史
        self.pruning_history.append(importance_penalty.item() if hasattr(importance_penalty, 'item') else 0.0)
        
        return base_loss + 0.1 * importance_penalty
    
    def get_layer_pruning_suggestions(self, current_pruning_ratio=0.5):
        """
        基于重要性权重给出剪枝建议
        """
        # 计算逆重要性（重要性低的层更容易被剪枝）
        inverse_importance = 1.0 / (self.importance_weights + 1e-8)
        
        # 计算建议剪枝的层数
        num_layers_to_prune = int(self.total_layers * current_pruning_ratio)
        
        # 按逆重要性排序，选择最不重要的层进行剪枝
        _, indices = torch.sort(inverse_importance, descending=True)
        suggested_prune_layers = indices[:num_layers_to_prune].tolist()
        
        return {
            'suggested_prune_layers': suggested_prune_layers,
            'layer_importance_scores': self.importance_weights.tolist(),
            'pruning_ratio': current_pruning_ratio,
            'num_layers_to_prune': num_layers_to_prune
        }
    
    def analyze_pruning_pattern(self, pruning_decisions):
        """
        分析当前剪枝模式的合理性
        """
        analysis = {
            'total_layers': self.total_layers,
            'pruning_decisions': [],
            'importance_violations': 0,
            'efficiency_score': 0.0
        }
        
        layer_idx = 0
        for decision in pruning_decisions:
            if isinstance(decision, torch.Tensor):
                probs = torch.softmax(decision, dim=-1)
                if len(probs.shape) > 1:
                    probs = torch.mean(probs, dim=0)  # 平均批次
                
                # 获取最可能的剪枝决策
                pruning_choice = torch.argmax(probs).item()
                
                # 检查是否违反重要性原则
                current_importance = self.importance_weights[layer_idx] if layer_idx < self.total_layers else 1.0
                
                if pruning_choice == 0 and current_importance > 1.2:  # 剪枝了重要层
                    analysis['importance_violations'] += 1
                
                analysis['pruning_decisions'].append({
                    'layer_idx': layer_idx,
                    'pruning_choice': pruning_choice,
                    'importance_weight': current_importance,
                    'decision_confidence': torch.max(probs).item()
                })
                
                layer_idx += 1
        
        # 计算效率评分
        total_importance_preserved = 0.0
        total_layers_preserved = 0
        
        for decision in analysis['pruning_decisions']:
            if decision['pruning_choice'] == 1:  # 保留层
                total_importance_preserved += decision['importance_weight']
                total_layers_preserved += 1
        
        if total_layers_preserved > 0:
            analysis['efficiency_score'] = total_importance_preserved / total_layers_preserved
        
        return analysis
    
    def get_statistics(self):
        """
        获取剪枝统计信息
        """
        return {
            'total_layers': self.total_layers,
            'importance_weights': self.importance_weights.tolist(),
            'avg_importance_penalty': np.mean(self.pruning_history) if self.pruning_history else 0.0,
            'penalty_history': self.pruning_history.copy(),
            'architecture': self.architecture
        }
