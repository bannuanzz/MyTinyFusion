"""
自适应温度调度器改进
解决Gumbel Softmax温度衰减过于简单的问题
"""
import torch
import torch.nn as nn
import numpy as np

class AdaptiveTemperatureScheduler:
    """
    基于决策置信度的自适应温度调度器
    """
    def __init__(self, initial_tau=5.0, min_tau=0.1, confidence_threshold=0.8):
        self.initial_tau = initial_tau
        self.min_tau = min_tau
        self.confidence_threshold = confidence_threshold
        self.tau = initial_tau
        self.confidence_history = []
        self.tau_history = []
        
    def update_temperature(self, gate_probs, epoch, total_epochs):
        """
        根据决策置信度动态调整温度
        """
        # 计算当前决策的平均置信度
        if isinstance(gate_probs, list):
            # 处理多个gate的情况
            all_probs = torch.cat([torch.softmax(gate.flatten(), dim=-1) for gate in gate_probs])
        else:
            all_probs = torch.softmax(gate_probs.flatten(), dim=-1)
            
        max_probs = torch.max(all_probs.view(-1, all_probs.size(-1)), dim=-1)[0]
        avg_confidence = torch.mean(max_probs).item()
        self.confidence_history.append(avg_confidence)
        
        # 计算置信度变化趋势
        confidence_trend = 0.0
        if len(self.confidence_history) > 5:
            recent_confidence = np.mean(self.confidence_history[-5:])
            older_confidence = np.mean(self.confidence_history[-10:-5]) if len(self.confidence_history) > 10 else recent_confidence
            confidence_trend = recent_confidence - older_confidence
        
        # 自适应调整策略
        if avg_confidence > self.confidence_threshold:
            # 高置信度时，加快温度降低速度
            if confidence_trend > 0.05:  # 置信度快速上升
                decay_factor = 0.92
            else:
                decay_factor = 0.95
        else:
            # 低置信度时，放慢温度降低速度
            if confidence_trend < -0.05:  # 置信度下降
                decay_factor = 0.99
            else:
                decay_factor = 0.98
            
        # 结合epoch进度的温度衰减
        progress = epoch / total_epochs
        
        # 非线性衰减：早期慢，中期快，后期稳定
        if progress < 0.3:
            # 早期：保持较高温度，充分探索
            base_decay = self.initial_tau * (1 - 0.2 * progress / 0.3)
        elif progress < 0.7:
            # 中期：快速衰减
            base_decay = self.initial_tau * 0.8 * (1 - (progress - 0.3) / 0.4)
        else:
            # 后期：缓慢衰减到最小值
            base_decay = self.initial_tau * 0.4 * (1 - (progress - 0.7) / 0.3)
        
        self.tau = max(
            self.min_tau, 
            base_decay * decay_factor
        )
        
        self.tau_history.append(self.tau)
        return self.tau
    
    def get_temperature(self):
        return self.tau
        
    def get_statistics(self):
        """
        获取调度器统计信息
        """
        return {
            'current_tau': self.tau,
            'avg_confidence': np.mean(self.confidence_history) if self.confidence_history else 0.0,
            'confidence_std': np.std(self.confidence_history) if len(self.confidence_history) > 1 else 0.0,
            'tau_reduction_rate': (self.initial_tau - self.tau) / self.initial_tau,
            'confidence_history': self.confidence_history.copy(),
            'tau_history': self.tau_history.copy()
        }
