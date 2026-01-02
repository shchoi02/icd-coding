import torch
import torch.nn as nn
import torch.nn.functional as F
from src.losses.utils import weight_reduce_loss
from src.losses.ce import binary_cross_entropy, cross_entropy, partial_cross_entropy
import numpy as np
from src.losses.mfm import MultiGrainedFocalLoss

  
    
class ResampleLoss2(nn.Module):
    """
    ICD용 수정 포인트:
        - SLP는 전체 C가 아니라 active subset(Ca)에서만 수행
        - logits/labels/prop/var/sigma/ro/tao는 모두 (Ca) 기준으로 들어온다고 가정
    """
    def __init__(
        self,
        class_instance_nums,
        up_mult=9, dw_mult=9,
        coef_alpha=1.0/3.0, coef_beta=0.7,
        use_slp=True,
        return_slp_debug=True,
        eps=1e-8,
        reduction="mean",
    ):
        super().__init__()
        self.use_slp = use_slp
        self.return_slp_debug = return_slp_debug
        self.up_mult = up_mult
        self.dw_mult = dw_mult
        self.coef_alpha = coef_alpha
        self.coef_beta = coef_beta
        self.eps = eps
        self.reduction = reduction

        class_freq = torch.tensor(class_instance_nums, dtype=torch.float32)
        self.register_buffer("class_freq", class_freq.clamp_min(1.0))

        # 원본 MFM 사용 (외부에서 주입해도 됨)
        self.mfm = MultiGrainedFocalLoss()
        self.mfm.create_weight(class_instance_nums)

    def forward(
        self,
        norm_prop, nonzero_var_tensor, zero_var_tensor,
        normalized_sigma_cj, normalized_ro_cj, normalized_tao_cj,
        cls_score, label,
        active_idx=None,
        return_slp_debug=False,
        **kwargs,
    ):
        """
        cls_score: [B, C_total] 또는 [B, Ca]
        label:     [B, C_total] 또는 [B, Ca]
        active_idx: ICD에서는 [Ca] (LongTensor) 주는 것을 권장
        """
        logits_raw_full = cls_score
        targets_full = label.float()

        if active_idx is not None:
            # (ICD) active subset으로 축소
            logits_raw = logits_raw_full.index_select(dim=1, index=active_idx)
            targets = targets_full.index_select(dim=1, index=active_idx)
        else:
            # (CXR) 그냥 전체 사용 (C가 작을 때만)
            logits_raw = logits_raw_full
            targets = targets_full

        if self.use_slp:
            logits_slp, dbg = self._apply_slp_subset(
                logits_raw, targets,
                norm_prop, nonzero_var_tensor, zero_var_tensor,
                normalized_sigma_cj, normalized_ro_cj, normalized_tao_cj,
                return_debug=True,
            )
        else:
            logits_slp = logits_raw
            dbg = {}

        loss = self.mfm(logits_slp, targets)

        if return_slp_debug or self.return_slp_debug:
            return loss, logits_raw.detach(), logits_slp.detach(), dbg
        return loss

    def _apply_slp_subset(
        self,
        logits_ca, labels_ca,
        prop_ca, nonzero_var_ca, zero_var_ca,
        sigma_ca, ro_ca, tao_ca,
        return_debug=False,
    ):
        """
        logits_ca:  [B, Ca]
        labels_ca:  [B, Ca]
        sigma/ro/tao: [Ca, Ca]
        """
        B, Ca = logits_ca.shape

        # 여기서만 [B, Ca, Ca]를 만듦. Ca는 수백~수천으로 제한해야 함.
        logits_exp = logits_ca.unsqueeze(-1).expand(B, Ca, Ca).clone()
        labels_exp = labels_ca.unsqueeze(-1).expand(B, Ca, Ca).clone()

        perturb, dbg_dict = self._lpl_imbalance(
            logits_exp, labels_exp,
            prop_ca, nonzero_var_ca, zero_var_ca,
            sigma_ca, ro_ca, tao_ca,
            return_debug=True,
        )
        logits_slp = (logits_exp + perturb).mean(dim=2)  # [B, Ca]

        if return_debug:
            return logits_slp, dbg_dict
        return logits_slp

    def _pgd_like_diff_sign(self, x, y, step, sign):
        y = y.to(torch.float32)
        iters = int(torch.max(step).item() + 1)
        logit = torch.zeros_like(x)
        for k in range(iters):
            grad = torch.sigmoid(x) - y
            x = x + grad * sign / x.shape[1]
            logit = logit + x * (step == k)
        return logit

    def _lpl_imbalance(
        self, logits, labels,
        prop, nonzero_var_tensor, zero_var_tensor,
        normalized_sigma_cj, normalized_ro_cj, normalized_tao_cj,
        return_debug=False,
    ):
        """
        logits, labels: [B, Ca, Ca]
        coef_cj: [Ca, Ca]
        """
        Ca = labels.size(1)
        eps = 1e-6

        safe_zero = torch.clamp(zero_var_tensor, min=eps)
        ratio = nonzero_var_tensor / safe_zero  # [Ca]

        lam  = self.coef_alpha
        beta = self.coef_beta

        coef_cc = (1.0 - (1.0 - lam) * beta) * prop + (1.0 - lam) * beta * ratio  # [Ca]
        coef_cj = lam * normalized_tao_cj + (1.0 - lam) * (beta * normalized_sigma_cj + (1.0 - beta) * normalized_ro_cj)  # [Ca,Ca]

        eye = torch.eye(Ca, device=logits.device, dtype=coef_cj.dtype)
        coef_cj = coef_cj * (1.0 - eye) + torch.diag(coef_cc)

        quant = coef_cj.mean()
        head_mask = (coef_cj > quant).to(coef_cj.dtype)
        tail_mask = 1.0 - head_mask

        def _group_minmax(x, mask):
            x_masked = x * mask
            if (mask > 0).sum() == 0:
                return x_masked
            valid = x_masked[mask > 0]
            v_min = valid.min()
            v_max = valid.max()
            scale = (v_max - v_min + eps)
            x_norm = (x_masked - v_min) / scale
            return torch.clamp(x_norm, min=0.0)

        head_coef = _group_minmax(coef_cj, head_mask)
        tail_coef = _group_minmax(coef_cj, tail_mask)

        head_dw_steps = torch.floor(head_coef * self.dw_mult)
        tail_up_steps = torch.floor(tail_coef * self.up_mult)

        logits_head_dw = self._pgd_like_diff_sign(logits, labels, head_dw_steps, -1.0) - logits
        logits_tail_up = self._pgd_like_diff_sign(logits, labels, tail_up_steps,  1.0) - logits

        perturb = (logits_head_dw + logits_tail_up).detach()

        if return_debug:
            return perturb, {
                "coef_cj": coef_cj.detach(),
                "head_dw_steps": head_dw_steps.detach(),
                "tail_up_steps": tail_up_steps.detach(),
            }
        return perturb
    

class ResampleLoss(nn.Module):

    def __init__(self,
                 up_mult=5,dw_mult=3,
                 use_sigmoid=False,
                 reduction='mean',
                 loss_weight=1.0,
                 partial=False,
                 focal=dict(
                     focal=False,
                     balance_param=2.0,
                     gamma=2,
                 ),
                 CB_loss=dict(
                     CB_beta=0.9,
                     CB_mode='average_w'  # 'by_class', 'average_n', 'average_w', 'min_n'
                 ),
                 map_param=dict(
                     alpha=10.0,
                     beta=0.2,
                     gamma=0.1
                 ),
                 logit_reg=dict(
                     neg_scale=5.0,
                     init_bias=0.1
                 ),
                 coef_param=dict(
                      coef_alpha=0.5,
                      coef_beta=0.5
        
                 ),
                 reweight_func=None,  # None, 'inv', 'sqrt_inv', 'rebalance', 'CB'
                 weight_norm=None, # None, 'by_instance', 'by_batch'
                 class_freq=None, neg_class_freq=None):
        super(ResampleLoss, self).__init__()

        assert (use_sigmoid is True) or (partial is False)
        self.up_mult = up_mult
        self.dw_mult = dw_mult
        self.use_sigmoid = use_sigmoid
        self.partial = partial
        self.loss_weight = loss_weight
        self.reduction = reduction
        if self.use_sigmoid:
            if self.partial:
                self.cls_criterion = partial_cross_entropy
            else:
                self.cls_criterion = binary_cross_entropy
        else:
            self.cls_criterion = cross_entropy

        # reweighting function
        self.reweight_func = reweight_func

        # normalization (optional)
        self.weight_norm = weight_norm

        # focal loss params
        self.focal = focal['focal']
        self.gamma = focal['gamma']
        self.balance_param = focal['balance_param']

        # mapping function params
        self.map_alpha = map_param['alpha']
        self.map_beta = map_param['beta']
        self.map_gamma = map_param['gamma']

        # CB loss params (optional)
        self.CB_beta = CB_loss['CB_beta']
        self.CB_mode = CB_loss['CB_mode']

        # coef params
        self.coef_alpha = coef_param['coef_alpha']
        self.coef_beta = coef_param['coef_beta']

        self.eps = 1e-8
        self.class_freq = class_freq.clamp_min(self.eps).cuda()
        self.neg_class_freq = neg_class_freq.clamp_min(self.eps).cuda()
        self.num_classes = self.class_freq.shape[0]
        self.train_num = (self.class_freq + self.neg_class_freq).mean()
        # regularization params
        self.logit_reg = logit_reg
        self.neg_scale = logit_reg[
            'neg_scale'] if 'neg_scale' in logit_reg else 1.0
        init_bias = logit_reg['init_bias'] if 'init_bias' in logit_reg else 0.0
        self.init_bias = - torch.log(
            self.train_num / self.class_freq - 1) * init_bias / self.neg_scale

        self.freq_inv = torch.ones(self.class_freq.shape).cuda() / self.class_freq
        self.propotion_inv = self.train_num / self.class_freq

        # print('\033[1;35m loading from {} | {} | {} | s\033[0;0m'.format(freq_file, reweight_func, logit_reg))
        # print('\033[1;35m rebalance reweighting mapping params: {:.2f} | {:.2f} | {:.2f} \033[0;0m'.format(self.map_alpha, self.map_beta, self.map_gamma))

    def forward(self,
                norm_prop, 
                nonzero_var_tensor, 
                zero_var_tensor, 
                normalized_sigma_cj, 
                normalized_ro_cj, 
                normalized_tao_cj,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        weight = self.reweight_functions(label)
        cls_score, weight = self.logit_reg_functions(label.float(), cls_score, norm_prop, nonzero_var_tensor, zero_var_tensor, normalized_sigma_cj, normalized_ro_cj, 
                normalized_tao_cj,weight)
      
        if self.focal:
            logpt = - self.cls_criterion(
                cls_score.clone(), label, weight=None, reduction='none',
                avg_factor=avg_factor)
            # pt is sigmoid(logit) for pos or sigmoid(-logit) for neg
            pt = torch.exp(logpt)
            loss = self.cls_criterion(
                cls_score, label.float(), weight=weight, reduction='none')
            loss = ((1 - pt) ** self.gamma) * loss
            loss = self.balance_param * loss
        else:
            loss = self.cls_criterion(cls_score, label.float(), weight,
                                      reduction=reduction)

        loss = self.loss_weight * loss
        return loss

    def reweight_functions(self, label):
        if self.reweight_func is None:
            return None
        elif self.reweight_func in ['inv', 'sqrt_inv']:
            weight = self.RW_weight(label.float())
        elif self.reweight_func in 'rebalance':
            weight = self.rebalance_weight(label.float())
        elif self.reweight_func in 'CB':
            weight = self.CB_weight(label.float())
        else:
            return None

        if self.weight_norm is not None:
            if 'by_instance' in self.weight_norm:
                max_by_instance, _ = torch.max(weight, dim=-1, keepdim=True)
                weight = weight / max_by_instance
            elif 'by_batch' in self.weight_norm:
                weight = weight / torch.max(weight)

        return weight

    def pgd_like(self, x, y,step,sign):
        y = y.to(torch.float32)
        iters = int(torch.max(step).item()+1)
        logit=torch.zeros_like(x)
        for k in range(iters):
            grad = torch.sigmoid(x)-y
            x = x + grad*sign/x.shape[1]
            logit = logit + x*(step==k)
        return logit

    def pgd_like_diff_sign(self, x, y, step, sign):


        y = y.to(torch.float32)

        iters = int(torch.max(step).item()+1)
        logit = torch.zeros_like(x)
        for k in range(iters):
            grad = torch.sigmoid(x)-y
            x = x + grad*sign/x.shape[1]
            logit = logit + x*(step==k)
        return logit

    def lpl(self,logits, labels):

        # compute split
        quant = self.train_num*0.5
        split = torch.where(self.class_freq>quant,1,0)

        # compute head bound  
        head_dw_steps = torch.ones_like(split)*self.dw_mult

        # compute tail bound
        max_tail = torch.max(self.class_freq*(1-split))
        tail_up_steps = torch.floor(-torch.log(self.class_freq/max_tail)+0.5)*self.up_mult

        logits_head_dw = self.pgd_like(logits, labels, head_dw_steps, -1.0) - logits   # 极小化（正头部，负尾部）
        logits_tail_up = self.pgd_like(logits, labels, tail_up_steps, 1.0) - logits    # 极大化 （正尾 ，负头）

        head = torch.sum(logits_head_dw*labels*split,dim=0)/(torch.sum(labels*split,dim=0)+1e-6)
        tail = torch.sum(logits_tail_up*labels*(1-split),dim=0)/(torch.sum(labels*(1-split),dim=0)+1e-6)

        # compute perturb
        perturb = head+tail

        return perturb.detach()


    def lpl_imbalance(self, logits_2d, labels_2d,
                  prop, nonzero_var_tensor, zero_var_tensor,
                  normalized_sigma_cj, normalized_ro_cj, normalized_tao_cj):
        """
        logits_2d: [B, C]  (GPU)
        labels_2d: [B, C]  (GPU)
        prop, nonzero_var_tensor, zero_var_tensor: [C] (GPU)
        normalized_*_cj: [C, C] (CPU!)  ← 중요
        return: perturb [B, C] (GPU)
        """
        device = logits_2d.device
        eps = 1e-8
        B, C = logits_2d.shape

        # 1) diag 계수 (GPU, 길이 C)
        coef_cc = (1 - (1 - self.coef_alpha) * self.coef_beta) * prop \
                + (1 - self.coef_alpha) * self.coef_beta * (nonzero_var_tensor / (zero_var_tensor + eps))  # [C], GPU

        # 2) CPU에서 행 평균(대각 제외)만 계산 → 길이 C 벡터
        def row_mean_exdiag_cpu(M_cpu: torch.Tensor) -> torch.Tensor:
            # M_cpu: [C, C] on CPU
            s = M_cpu.sum(dim=1) - M_cpu.diag()
            return s / max(C - 1, 1)

        mean_sigma = row_mean_exdiag_cpu(normalized_sigma_cj)  # [C], CPU
        mean_ro    = row_mean_exdiag_cpu(normalized_ro_cj)     # [C], CPU
        mean_tao   = row_mean_exdiag_cpu(normalized_tao_cj)    # [C], CPU

        coef_off_cpu = self.coef_alpha * mean_tao + \
                    (1 - self.coef_alpha) * (self.coef_beta * mean_sigma + (1 - self.coef_beta) * mean_ro)  # [C], CPU

        # 3) 최종 per-class 계수만 GPU로
        coef_i = (coef_off_cpu.to(device, dtype=logits_2d.dtype) + coef_cc).clamp_min(0)  # [C], GPU

        # 4) split & steps (길이 C, GPU)
        thresh = coef_i.mean()
        head_coef = torch.where(coef_i > thresh, coef_i, torch.zeros_like(coef_i))  # [C]
        tail_coef = torch.where(coef_i <= thresh, coef_i, torch.zeros_like(coef_i)) # [C]

        head_dw_steps = torch.floor(head_coef * self.dw_mult)  # [C]
        tail_up_steps = torch.floor(tail_coef * self.up_mult)  # [C]

        # 5) 2D PGD-like 업데이트
        def pgd_like_2d(x, y, steps, sign):
            iters = int(steps.max().item()) if steps.numel() else 0
            if iters <= 0:
                return x
            xk = x
            for k in range(iters):
                mask = (steps > k).float().view(1, C)  # [1, C]
                grad = torch.sigmoid(xk) - y
                xk   = xk + (grad * sign / C) * mask
            return xk

        logits_head_dw = pgd_like_2d(logits_2d, labels_2d, head_dw_steps, sign=-1.0) - logits_2d
        logits_tail_up = pgd_like_2d(logits_2d, labels_2d, tail_up_steps, sign=+1.0) - logits_2d
        perturb = logits_head_dw + logits_tail_up  # [B, C]
        return perturb.detach()


    def logit_reg_functions(self, labels, logits,
                        norm_prop, nonzero_var_tensor, zero_var_tensor,
                        normalized_sigma_cj, normalized_ro_cj, normalized_tao_cj,
                        weight=None):
        if not self.logit_reg:
            return logits, weight
        B, C = logits.shape

        if 'init_bias' in self.logit_reg:
            perturb = self.lpl_imbalance(logits, labels, norm_prop,
                                        nonzero_var_tensor, zero_var_tensor,
                                        normalized_sigma_cj, normalized_ro_cj, normalized_tao_cj)  # [B,C]
            logits = logits + perturb

        if 'neg_scale' in self.logit_reg:
            logits = logits * (1 - labels) * self.neg_scale + logits * labels
            if weight is not None:
                weight = weight / self.neg_scale * (1 - labels) + weight * labels

        return logits, weight

    def rebalance_weight(self, gt_labels):
        repeat_rate = torch.sum( gt_labels.float() * self.freq_inv, dim=1, keepdim=True)
        pos_weight = self.freq_inv.clone().detach().unsqueeze(0) / repeat_rate
        # pos and neg are equally treated
        weight = torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + self.map_alpha
        return weight

    def CB_weight(self, gt_labels):
        if  'by_class' in self.CB_mode:
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, self.class_freq)).cuda()
        elif 'average_n' in self.CB_mode:
            avg_n = torch.sum(gt_labels * self.class_freq, dim=1, keepdim=True) / \
                    torch.sum(gt_labels, dim=1, keepdim=True)
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, avg_n)).cuda()
        elif 'average_w' in self.CB_mode:
            weight_ = torch.tensor((1 - self.CB_beta)).cuda() / \
                      (1 - torch.pow(self.CB_beta, self.class_freq)).cuda()
            weight = torch.sum(gt_labels * weight_, dim=1, keepdim=True) / \
                     torch.sum(gt_labels, dim=1, keepdim=True)
        elif 'min_n' in self.CB_mode:
            min_n, _ = torch.min(gt_labels * self.class_freq +
                                 (1 - gt_labels) * 100000, dim=1, keepdim=True)
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, min_n)).cuda()
        else:
            raise NameError
        return weight

    def RW_weight(self, gt_labels, by_class=True):
        if 'sqrt' in self.reweight_func:
            weight = torch.sqrt(self.propotion_inv)
        else:
            weight = self.propotion_inv
        if not by_class:
            sum_ = torch.sum(weight * gt_labels, dim=1, keepdim=True)
            weight = sum_ / torch.sum(gt_labels, dim=1, keepdim=True)
        return weight

    def data_normal(self, data):
        d_min = torch.min(data)
        d_max = torch.max(data)
        dst = d_max-d_min
        norm_data = torch.div(data-d_min,dst)
        reverse_norm_data = torch.div(d_max-data,dst)
        return norm_data, reverse_norm_data

    def none_zero_normal(self, data):
        ones = torch.ones_like(data)
        d_min = torch.min(torch.where(data==0,ones,data))
        d_max = torch.max(data)
        dst = d_max - d_min
        norm_data =torch.div(data-d_min,dst)
        norm_data =torch.clamp(norm_data,min=0.0)
        reverse_norm_data =torch.div(d_max-data,dst)
        zero = torch.zeros_like(reverse_norm_data)
        reverse_norm_data =torch.where(data>1,zero,data)
        return norm_data, reverse_norm_data
