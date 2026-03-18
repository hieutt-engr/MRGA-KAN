import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha # Tensor class weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss) # prob correct
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class PolyFocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=None, gamma=2.0, epsilon=1.0, label_smoothing=0.1, reduction='mean'):
        """
        Args:
            alpha: Class weights (Tensor).
            gamma: Focal parameter (thường là 2.0).
            epsilon: Tham số của Poly1 (thường là 1.0). Giúp loss ổn định hơn Focal thuần.
            label_smoothing: Mức độ làm trơn nhãn (0.05 - 0.1).
        """
        super(PolyFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, target):
        with torch.no_grad():
            target_one_hot = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1)
            # Công thức Label Smoothing: (1 - LS) * y + LS / K
            smooth_target = target_one_hot * (1.0 - self.label_smoothing) + \
                            (self.label_smoothing / self.num_classes)
        
        probs = torch.softmax(logits, dim=-1)
        pt = (probs * target_one_hot).sum(dim=1)
        
        log_probs = torch.log_softmax(logits, dim=-1)
        ce_loss = -(smooth_target * log_probs).sum(dim=1)
        
        if self.alpha is not None:
            weights = self.alpha[target]
            ce_loss = ce_loss * weights
            
        focal_term = (1 - pt).pow(self.gamma)
        poly1_loss = focal_term * ce_loss + self.epsilon * (1 - pt)
        
        if self.reduction == 'mean':
            return poly1_loss.mean()
        elif self.reduction == 'sum':
            return poly1_loss.sum()
        else:
            return poly1_loss
        
class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        """
        cls_num_list: List chứa số lượng mẫu của từng class (để tự tính margin).
        max_m: Margin tối đa (thường 0.5).
        s: Scale factor (thường 30).
        """
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        
        # Cộng Margin vào đúng vị trí class ground-truth
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m # Trừ đi margin (làm khó mô hình hơn)
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)