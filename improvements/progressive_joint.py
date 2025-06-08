"""
渐进式联合优化改进
解决剪枝和恢复阶段割裂的问题
"""
import torch
import torch.nn as nn

class ProgressiveJointOptimization:
    """
    渐进式联合优化框架
    同时进行剪枝决策学习和知识蒸馏
    """
    def __init__(self, student_model, teacher_model, alpha=0.7):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.alpha = alpha  # 剪枝损失权重
        self.beta = 1 - alpha  # 蒸馏损失权重
        
    def compute_joint_loss(self, x, t, y, epoch, total_epochs):
        """
        计算联合损失：剪枝损失 + 蒸馏损失
        """
        # 获取教师模型输出（固定）
        with torch.no_grad():
            teacher_output = self.teacher_model(x, t, y)
            teacher_features = self.teacher_model.get_intermediate_features(x, t, y)
        
        # 学生模型前向传播（包含剪枝决策）
        student_output = self.student_model(x, t, y)
        student_features = self.student_model.get_intermediate_features(x, t, y)
        
        # 1. 基础重构损失
        reconstruction_loss = nn.MSELoss()(student_output, teacher_output)
        
        # 2. 特征蒸馏损失
        feature_distill_loss = self._compute_feature_distillation_loss(
            student_features, teacher_features
        )
        
        # 3. 剪枝正则化损失
        pruning_reg_loss = self._compute_pruning_regularization()
        
        # 4. 动态权重调整
        progress = epoch / total_epochs
        # 早期更注重剪枝，后期更注重蒸馏
        current_alpha = self.alpha * (1 - progress * 0.5)
        current_beta = self.beta * (1 + progress * 0.5)
        
        total_loss = (
            current_alpha * (reconstruction_loss + pruning_reg_loss) +
            current_beta * feature_distill_loss
        )
        
        return total_loss, {
            'reconstruction': reconstruction_loss.item(),
            'feature_distill': feature_distill_loss.item(),
            'pruning_reg': pruning_reg_loss.item(),
            'alpha': current_alpha,
            'beta': current_beta
        }
    
    def _compute_feature_distillation_loss(self, student_features, teacher_features):
        """
        计算特征蒸馏损失，处理维度不匹配问题
        """
        distill_loss = 0.0
        
        # 对齐学生和教师的特征层
        teacher_indices = self._get_teacher_alignment(len(student_features))
        
        for i, teacher_idx in enumerate(teacher_indices):
            if i < len(student_features) and teacher_idx < len(teacher_features):
                s_feat = student_features[i]
                t_feat = teacher_features[teacher_idx]
                
                # 特征维度对齐
                if s_feat.shape != t_feat.shape:
                    # 使用投影层或池化进行维度匹配
                    t_feat = self._align_features(s_feat, t_feat)
                
                distill_loss += nn.MSELoss()(s_feat, t_feat)
        
        return distill_loss / len(student_features)
    
    def _get_teacher_alignment(self, num_student_layers):
        """
        计算学生层与教师层的对应关系
        """
        total_teacher_layers = 28  # DiT-XL原始层数
        indices = []
        for i in range(num_student_layers):
            # 均匀映射
            teacher_idx = int(i * total_teacher_layers / num_student_layers)
            indices.append(teacher_idx)
        return indices
    
    def _align_features(self, student_feat, teacher_feat):
        """
        对齐特征维度
        """
        if student_feat.shape[-1] != teacher_feat.shape[-1]:
            # 使用平均池化降维
            pool_size = teacher_feat.shape[-1] // student_feat.shape[-1]
            teacher_feat = nn.AvgPool1d(pool_size)(teacher_feat.transpose(-1, -2)).transpose(-1, -2)
        return teacher_feat
    
    def _compute_pruning_regularization(self):
        """
        计算剪枝正则化损失，鼓励稀疏且一致的决策
        """
        reg_loss = 0.0
        
        for gate in self.student_model.gumbel_gates:
            # 鼓励决策的稀疏性
            probs = torch.softmax(gate * self.student_model.scaling, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            reg_loss += torch.mean(entropy)  # 鼓励低熵（稀疏决策）
        
        return reg_loss
