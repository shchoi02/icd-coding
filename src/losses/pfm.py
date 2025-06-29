import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from loss.SpccLoss import SPLC 

def build_attention_scores(y: torch.Tensor,
                           co_mat: torch.Tensor,
                           eps: float = 1e-8) -> torch.Tensor:

    y = y.to(co_mat.device)
    if y.dtype != torch.float32 and y.dtype != torch.float16:
        y = y.float()

    pos_cnt  = y.sum(dim=1, keepdim=True) 
    safe_cnt = pos_cnt.clamp(min=1) 
    att_sum  = y @ co_mat
    att_avg  = att_sum / safe_cnt
    att_avg  = att_avg * (pos_cnt > 0)
    row_sum  = att_avg.sum(dim=1, keepdim=True) 
    att_norm = att_avg / row_sum.clamp(min=eps) 

    return att_norm   


class PriorFocalModifierLoss(nn.Module):
    def __init__(self, gamma_neg=3, gamma_pos=1, gamma_class_ng=1.2, clip=0.05, eps=1e-8, \
            disable_torch_grad_focal_loss=True, distribution_path=None, co_occurrence_matrix=None):
        super(PriorFocalModifierLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_class_ng=gamma_class_ng
        self.gamma_class_pos=1
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.distribution_path=distribution_path
        
    @torch.no_grad()                  
    def create_weight(self, distribution):
        dist = torch.as_tensor(distribution, dtype=torch.float32, device='cuda')
        total = dist.sum()
        prob = dist / total
        prob = prob / (prob.max() + self.eps) 
        weight = torch.pow(-torch.log(prob.clamp_min(self.eps)) + 1.0, 1.0 / 6)
        self.weight = weight.cuda().detach()
        
    
    @torch.no_grad()    
    def create_co_occurrence_matrix(self, counts):
        mat = torch.as_tensor(counts, device="cuda")
        probs = mat / (mat.sum(dim=0, keepdim=True).clamp_min(1e-6))
        self.register_buffer("co_occurrence_matrix", probs)

    
    def forward(self, x, y):   
        
        weight = self.weight.to(dtype=x.dtype, device=x.device)    

        attention_scores_total=[]
        for k in range(y.shape[0]): 
            attention_scores=self.co_occurrence_matrix[y[k]==1].mean(dim=0)
            attention_scores=attention_scores/attention_scores.sum()
            attention_scores_total.append(attention_scores)
        final_attention_scores=torch.stack(attention_scores_total,0)
        # print(final_attention_scores)
                 
        # postive -
        x_sigmoid = torch.pow(torch.sigmoid(x),1) 
        # gamma_class_pos=self.gamma_class_pos
        # print (gamma_class_pos)
        gamma_class_pos=self.gamma_class_pos-final_attention_scores
        # gamma_class_pos=1      
        xs_pos = x_sigmoid*gamma_class_pos
        xs_neg = 1 - x_sigmoid
        
        # negtive +
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        xs_neg=torch.where(final_attention_scores==0, xs_neg, xs_neg*self.gamma_class_ng).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        loss*=weight

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:         
            with torch.no_grad():
                pt0 = xs_pos * y
                pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
                pt = pt0 + pt1
                one_sided_gamma = (self.gamma_pos)* y + (self.gamma_neg+weight)* (1 - y)
                one_sided_w = torch.pow(1 - pt, one_sided_gamma)           
            loss *= one_sided_w
        loss=-loss.sum()
        return loss

def create_loss(gamma_neg=3, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True):
    return PriorFocalModifierLoss(gamma_neg=3, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
