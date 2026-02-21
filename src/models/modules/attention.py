import math
import torch
import torch.nn as nn

from torch.nn.init import xavier_uniform_
import torch.nn.functional as F

class _CausalNormBase(nn.Module):
    def __init__(self, num_classes, feat_dim, use_effect=True, num_head=1, tau=16.0, alpha=2.0, gamma=0.03125):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, feat_dim), requires_grad=True)
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        self.num_head = int(num_head)
        self.head_dim = feat_dim // self.num_head

        self.scale = float(tau) / float(self.num_head)
        self.norm_scale = float(gamma)
        self.alpha = float(alpha)
        self.use_effect = bool(use_effect)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        with torch.no_grad():
            self.weight.uniform_(-stdv, stdv)

    def l2_norm(self, x):
        return x / (torch.norm(x, 2, 1, keepdim=True) + 1e-12)

    def causal_norm(self, x, weight):
        norm = torch.norm(x, 2, 1, keepdim=True)
        return x / (norm + weight)

    def get_cos(self, x, y):
        # x,y: [B,D]
        return (x * y).sum(-1, keepdim=True) / (
            (torch.norm(x, 2, 1, keepdim=True) + 1e-12) * (torch.norm(y, 2, 1, keepdim=True) + 1e-12)
        )

    def multi_head_call(self, func, x, weight=None):
        assert x.dim() == 2
        xs = torch.split(x, self.head_dim, dim=1)
        if weight is not None:
            ys = [func(t, weight) for t in xs]
        else:
            ys = [func(t) for t in xs]
        return torch.cat(ys, dim=1)

class BalancedCausalNormVec(_CausalNormBase):
    """x: [B,D] -> logits: [B,C]"""
    def forward(self, x, embed=None):
        # x: [B,D]
        normed_w = self.multi_head_call(self.causal_norm, self.weight, weight=self.norm_scale)  # [C,D]
        normed_x = self.multi_head_call(self.l2_norm, x)                                       # [B,D]
        y = torch.mm(normed_x * self.scale, normed_w.t())                                      # [B,C]
        y_nomoving = y.clone()
        return y, y_nomoving
    
class BalancedCausalNormDiagonal(_CausalNormBase):
    """
    입력: pooled = weighted_output [B, C, D]  (LAAT pooled rep)
    출력: diag logits [B, C]
      - 각 클래스 c는 pooled[:,c,:] 와 weight[c] 로만 점수화 (third_linear 대체와 동일한 형태)
    """
    def forward(self, pooled, embed=None):
        # pooled: [B,C,D]
        B, C, D = pooled.shape
        assert C == self.num_classes and D == self.feat_dim

        # normed_w: [C,D]
        normed_w = self.multi_head_call(self.causal_norm, self.weight, weight=self.norm_scale)

        # normed_x: [B,C,D]
        x = pooled.reshape(B * C, D)
        normed_x = self.multi_head_call(self.l2_norm, x).reshape(B, C, D)

        # diag dot: (B,C,D) · (C,D) -> (B,C)
        y = (normed_x * normed_w.unsqueeze(0)).sum(dim=-1) * self.scale
        y_nomoving = y.clone()
        return y, y_nomoving

class HeadCausalNormDiagonal(_CausalNormBase):
    """
    입력: pooled [B,C,D], embed [D]
    출력: [B,C] (diag) + nomoving(효과 적용 diag)
    """
    def forward(self, pooled, embed):
        B, C, D = pooled.shape
        assert C == self.num_classes and D == self.feat_dim

        normed_w = self.multi_head_call(self.causal_norm, self.weight, weight=self.norm_scale)

        x = pooled.reshape(B * C, D)
        normed_x = self.multi_head_call(self.l2_norm, x).reshape(B, C, D)

        y = (normed_x * normed_w.unsqueeze(0)).sum(dim=-1) * self.scale
        y_nomoving = y.clone()

        if self.use_effect and embed is not None:
            c = embed.view(1, -1).to(pooled.device, dtype=pooled.dtype)  # [1,D]
            normed_c = self.multi_head_call(self.l2_norm, c)             # [1,D]

            # head-wise split (dim=-1)
            x_list = torch.split(normed_x, self.head_dim, dim=2)         # list of [B,C,dh]
            c_list = torch.split(normed_c, self.head_dim, dim=1)         # list of [1,dh]
            w_list = torch.split(normed_w, self.head_dim, dim=1)         # list of [C,dh]

            outs = []
            for nx, nc, nw in zip(x_list, c_list, w_list):
                # cos(nx, nc): [B,C,dh] vs [1,dh] -> [B,C,1]
                cos_val = (nx * nc.unsqueeze(1)).sum(-1, keepdim=True) / (
                    (torch.norm(nx, 2, dim=-1, keepdim=True) + 1e-12) *
                    (torch.norm(nc, 2, dim=-1, keepdim=True).unsqueeze(0).unsqueeze(0) + 1e-12)
                )
                nx_eff = nx - cos_val * nc.unsqueeze(1)  # [B,C,dh]

                # diag dot with nw: [B,C,dh] · [C,dh] -> [B,C]
                y_temp = (nx_eff * nw.unsqueeze(0)).sum(-1) * self.scale
                outs.append(y_temp)

            y_nomoving = sum(outs)

        return y, y_nomoving

class TailCausalNormDiagonal(_CausalNormBase):
    def forward(self, pooled, embed):
        B, C, D = pooled.shape
        assert C == self.num_classes and D == self.feat_dim

        normed_w = self.multi_head_call(self.causal_norm, self.weight, weight=self.norm_scale)

        x = pooled.reshape(B * C, D)
        normed_x = self.multi_head_call(self.l2_norm, x).reshape(B, C, D)

        y = (normed_x * normed_w.unsqueeze(0)).sum(dim=-1) * self.scale
        y_nomoving = y.clone()

        if self.use_effect and embed is not None:
            c = embed.view(1, -1).to(pooled.device, dtype=pooled.dtype)  # [1,D]
            normed_c = self.multi_head_call(self.l2_norm, c)             # [1,D]

            x_list = torch.split(normed_x, self.head_dim, dim=2)
            c_list = torch.split(normed_c, self.head_dim, dim=1)
            w_list = torch.split(normed_w, self.head_dim, dim=1)

            outs = []
            for nx, nc, nw in zip(x_list, c_list, w_list):
                cos_val = (nx * nc.unsqueeze(1)).sum(-1, keepdim=True) / (
                    (torch.norm(nx, 2, dim=-1, keepdim=True) + 1e-12) *
                    (torch.norm(nc, 2, dim=-1, keepdim=True).unsqueeze(0).unsqueeze(0) + 1e-12)
                )
                nx_eff = nx + cos_val * self.alpha * nc.unsqueeze(1)
                y_temp = (nx_eff * nw.unsqueeze(0)).sum(-1) * self.scale
                outs.append(y_temp)

            y_nomoving = sum(outs)

        return y, y_nomoving


class LabelAttentionHead(nn.Module):
    def __init__(self, input_size, projection_size, num_classes, num_head=1):
        super().__init__()
        self.first_linear = nn.Linear(input_size, projection_size, bias=False)
        self.second_linear = nn.Linear(projection_size, num_classes, bias=False)
        self._init_weights(0.0, 0.03)

        self.classifier = HeadCausalNormDiagonal(
            num_classes=num_classes,
            feat_dim=input_size,
            use_effect=False, # 수정
            num_head=num_head
        )

    def forward(self, x, attention_mask_1d=None, embed=None):
        weights = torch.tanh(self.first_linear(x))
        att_logits = self.second_linear(weights)

        if attention_mask_1d is not None:
            att_logits = att_logits.masked_fill(attention_mask_1d.unsqueeze(-1).eq(0), float("-inf"))

        att = F.softmax(att_logits, dim=1).transpose(1, 2)  # [B,C,L]
        pooled = att @ x                                     # [B,C,H]

        z, z_nm = self.classifier(pooled, embed)
        f = pooled.mean(dim=1)
        return z, z_nm, f

    def _init_weights(self, mean=0.0, std=0.03):
        nn.init.normal_(self.first_linear.weight, mean, std)
        nn.init.normal_(self.second_linear.weight, mean, std)

class LabelAttentionBalanced(nn.Module):
    def __init__(self, input_size, projection_size, num_classes, num_head=1):
        super().__init__()
        self.first_linear = nn.Linear(input_size, projection_size, bias=False)
        self.second_linear = nn.Linear(projection_size, num_classes, bias=False)
        self._init_weights(0.0, 0.03)

        self.classifier = BalancedCausalNormDiagonal(
            num_classes=num_classes,
            feat_dim=input_size,
            use_effect=False,
            num_head=num_head
        )

    def forward(self, x, attention_mask_1d=None, embed=None):
        weights = torch.tanh(self.first_linear(x))
        att_logits = self.second_linear(weights)

        if attention_mask_1d is not None:
            att_logits = att_logits.masked_fill(attention_mask_1d.unsqueeze(-1).eq(0), float("-inf"))

        att = F.softmax(att_logits, dim=1).transpose(1, 2)  # [B,C,L]
        pooled = att @ x                                     # [B,C,H]

        z, z_nm = self.classifier(pooled, embed)
        f = pooled.mean(dim=1)
        return z, z_nm, f

    def _init_weights(self, mean=0.0, std=0.03):
        nn.init.normal_(self.first_linear.weight, mean, std)
        nn.init.normal_(self.second_linear.weight, mean, std)


class LabelAttentionTail(nn.Module):
    def __init__(self, input_size, projection_size, num_classes, num_head=1):
        super().__init__()
        self.first_linear = nn.Linear(input_size, projection_size, bias=False)
        self.second_linear = nn.Linear(projection_size, num_classes, bias=False)
        self._init_weights(0.0, 0.03)

        self.classifier = TailCausalNormDiagonal(
            num_classes=num_classes,
            feat_dim=input_size,
            use_effect=False, # 수정
            num_head=num_head
        )

    def forward(self, x, attention_mask_1d=None, embed=None):
        weights = torch.tanh(self.first_linear(x))
        att_logits = self.second_linear(weights)

        if attention_mask_1d is not None:
            att_logits = att_logits.masked_fill(attention_mask_1d.unsqueeze(-1).eq(0), float("-inf"))

        att = F.softmax(att_logits, dim=1).transpose(1, 2)  # [B,C,L]
        pooled = att @ x                                     # [B,C,H]

        z, z_nm = self.classifier(pooled, embed)
        f = pooled.mean(dim=1)
        return z, z_nm, f

    def _init_weights(self, mean=0.0, std=0.03):
        nn.init.normal_(self.first_linear.weight, mean, std)
        nn.init.normal_(self.second_linear.weight, mean, std)
         
    
class LabelAttention(nn.Module):
    def __init__(self, input_size: int, projection_size: int, num_classes: int):
        super().__init__()
        self.first_linear = nn.Linear(input_size, projection_size, bias=False)
        self.second_linear = nn.Linear(projection_size, num_classes, bias=False)
        self.third_linear = nn.Linear(input_size, num_classes)
        self._init_weights(mean=0.0, std=0.03)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LAAT attention mechanism

        Args:
            x (torch.Tensor): [batch_size, seq_len, input_size]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """
        weights = torch.tanh(self.first_linear(x))
        att_weights = self.second_linear(weights)
        att_weights = torch.nn.functional.softmax(att_weights, dim=1).transpose(1, 2)
        weighted_output = att_weights @ x
        return (
            self.third_linear.weight.mul(weighted_output)
            .sum(dim=2)
            .add(self.third_linear.bias)
        )

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        """
        Initialise the weights

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.03.
        """

        torch.nn.init.normal_(self.first_linear.weight, mean, std)
        torch.nn.init.normal_(self.second_linear.weight, mean, std)
        torch.nn.init.normal_(self.third_linear.weight, mean, std)


class CAMLAttention(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.first_linear = nn.Linear(input_size, num_classes)
        xavier_uniform_(self.first_linear.weight)
        self.second_linear = nn.Linear(input_size, num_classes)
        xavier_uniform_(self.second_linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """CAML attention mechanism

        Args:
            x (torch.Tensor): [batch_size, input_size, seq_len]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """
        x = torch.tanh(x)
        weights = torch.softmax(self.first_linear.weight.matmul(x), dim=2)
        weighted_output = weights @ x.transpose(1, 2)
        return (
            self.second_linear.weight.mul(weighted_output)
            .sum(2)
            .add(self.second_linear.bias)
        )
