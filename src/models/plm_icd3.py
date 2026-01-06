# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch RoBERTa model. """
import torch, math, copy
import torch.utils.checkpoint
from torch import nn
from typing import Optional

from transformers import RobertaModel, AutoConfig

from src.models.modules.attention import LabelAttention
import torch.nn.functional as F
from src.losses.asl import ASLwithClassWeight


class BranchAdapter(nn.Module):
    """
    strong adapter: LN -> (D->r->D) -> residual
    """
    def __init__(self, dim: int, r: int = 256, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, r, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(r, dim, bias=False)
        nn.init.zeros_(self.fc2.weight)

    def forward(self, x):
        h = self.ln(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        return x + h

# =========================================================
# masked_softmax
# - COMIC Eq8에서 q=1, k=2 고정이면 padding 개념이 없으니
#   그냥 마지막 dim softmax면 충분
# =========================================================
def masked_softmax(X, valid_lens=None):
    return F.softmax(X, dim=-1)

# =========================================================
# (Eq8) Additive Attention (원본 레포 스타일: AdditiveAttention)
#   - q: [B,1,D], k/v: [B,2,D]
# =========================================================
class AdditivetionAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout):
        super().__init__()
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        # queries: [B, q, dq], keys/values: [B, k, dk]
        queries = self.W_q(queries)   # [B,q,H]
        keys    = self.W_k(keys)      # [B,k,H]
        features = torch.tanh(queries.unsqueeze(2) + keys.unsqueeze(1))  # [B,q,k,H]
        scores   = self.w_v(features).squeeze(-1)                        # [B,q,k]
        attn     = masked_softmax(scores, valid_lens)                    # [B,q,k]
        return torch.bmm(self.dropout(attn), values)                     # [B,q,dv]


class AdditiveEnvAttention(nn.Module):
    """
    Eq.(8): f_b = f_hat_b + 0.1 * Attn(f_hat_b, [f_h, f_t])
    """
    def __init__(self, dim=768, num_hiddens=768, dropout=0.1, attn_scale=0.1):
        super().__init__()
        self.attn = AdditivetionAttention(dim, dim, num_hiddens, dropout)
        self.attn_scale = attn_scale

    def forward(self, f_hat_b, f_h, f_t):
        # f_hat_b,f_h,f_t: [B,D]
        q  = f_hat_b.unsqueeze(1)            # [B,1,D]
        kv = torch.stack([f_h, f_t], dim=1)  # [B,2,D]
        ctx = self.attn(q, kv, kv).squeeze(1)  # [B,D]
        return f_hat_b + self.attn_scale * ctx
    
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


class BalancedCausalNormClassifier(_CausalNormBase):
    """
    balanced: y_b = (normed_x * scale) @ normed_w^T
    (네가 올린 balanced 버전: effect block이 주석 처리된 형태)
    """
    def forward(self, x, embed=None):
        # x: [B,D], embed: [D] or [1,D] (있어도 안 씀)
        normed_w = self.multi_head_call(self.causal_norm, self.weight, weight=self.norm_scale)
        normed_x = self.multi_head_call(self.l2_norm, x)
        y_b = torch.mm(normed_x * self.scale, normed_w.t())
        y_b_nomoving = y_b.clone()
        return y_b, y_b_nomoving


class HeadCausalNormClassifier(_CausalNormBase):
    """
    head: y_head_nomoving = sum( (nx - cos*nc) * scale @ nw^T )
    """
    def forward(self, x, embed):
        normed_w = self.multi_head_call(self.causal_norm, self.weight, weight=self.norm_scale)
        normed_x = self.multi_head_call(self.l2_norm, x)
        y_head = torch.mm(normed_x * self.scale, normed_w.t())
        y_head_nomoving = y_head.clone()

        if self.use_effect and embed is not None:
            if isinstance(embed, torch.Tensor):
                c = embed.view(1, -1).to(x.device, dtype=x.dtype)
            else:
                raise TypeError("embed must be torch.Tensor")

            normed_c = self.multi_head_call(self.l2_norm, c)  # [1,D]
            x_list = torch.split(normed_x, self.head_dim, dim=1)
            c_list = torch.split(normed_c, self.head_dim, dim=1)
            w_list = torch.split(normed_w, self.head_dim, dim=1)

            outs = []
            for nx, nc, nw in zip(x_list, c_list, w_list):
                cos_val = self.get_cos(nx, nc)  # [B,1]
                y_temp = torch.mm((nx - cos_val * nc) * self.scale, nw.t())
                outs.append(y_temp)
            y_head_nomoving = sum(outs)

        return y_head, y_head_nomoving


class TailCausalNormClassifier(_CausalNormBase):
    """
    tail: y_tail_nomoving = sum( (nx + alpha*cos*nc) * scale @ nw^T )
    """
    def forward(self, x, embed):
        normed_w = self.multi_head_call(self.causal_norm, self.weight, weight=self.norm_scale)
        normed_x = self.multi_head_call(self.l2_norm, x)
        y_tail = torch.mm(normed_x * self.scale, normed_w.t())
        y_tail_nomoving = y_tail.clone()

        if self.use_effect and embed is not None:
            if isinstance(embed, torch.Tensor):
                c = embed.view(1, -1).to(x.device, dtype=x.dtype)
            else:
                raise TypeError("embed must be torch.Tensor")

            normed_c = self.multi_head_call(self.l2_norm, c)  # [1,D]
            x_list = torch.split(normed_x, self.head_dim, dim=1)
            c_list = torch.split(normed_c, self.head_dim, dim=1)
            w_list = torch.split(normed_w, self.head_dim, dim=1)

            outs = []
            for nx, nc, nw in zip(x_list, c_list, w_list):
                cos_val = self.get_cos(nx, nc)  # [B,1]
                y_temp = torch.mm((nx + cos_val * self.alpha * nc) * self.scale, nw.t())
                outs.append(y_temp)
            y_tail_nomoving = sum(outs)

        return y_tail, y_tail_nomoving
    

class HeadTailBalancerLoss(nn.Module):
    def __init__(self, gamma=2, PFM=None):
        super(HeadTailBalancerLoss, self).__init__()
        self.gamma = gamma
        self.PFM = PFM
        self.eps = 1e-8

    def forward(self, head, tail, balance, labels):
        labels = labels.float()

        with torch.no_grad():
            h_acc = self.PFM(head, labels).pow(self.gamma)
            t_acc = self.PFM(tail, labels).pow(self.gamma)
            denom = h_acc + t_acc + self.eps
            k_h, k_t = h_acc / denom, t_acc / denom
            
        p_h = F.softmax(head, dim=-1)
        p_t = F.softmax(tail, dim=-1)
        p_b = F.softmax(balance, dim=-1)

        loss_h = self.PFM(p_h * p_b, labels)            
        loss_t = self.PFM(p_t * p_b, labels)

        loss = (k_h * loss_h + k_t * loss_t).mean()
        return loss
 

def masked_mean_pool(hidden: torch.Tensor, mask1d: Optional[torch.Tensor]):
    # hidden: (B, L, H), mask1d: (B, L) with 1 valid / 0 pad
    if mask1d is None:
        return hidden.mean(dim=1)
    mask = mask1d.unsqueeze(-1).to(hidden.dtype)          # (B,L,1)
    denom = mask.sum(dim=1).clamp(min=1.0)                # (B,1)
    return (hidden * mask).sum(dim=1) / denom             # (B,H)


class PLMICD3(nn.Module):
    def __init__(self, num_classes: int, model_path: str,
                 cls_num_list = None, 
                 head_idx = None, tail_idx = None,
                 co_occurrence_matrix = None,
                 class_freq = None, neg_class_freq = None,
                 **kwargs):
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            model_path, num_labels=num_classes, finetuning_task=None
        )
        self.register_buffer("head_idx", torch.tensor(head_idx))
        self.register_buffer("tail_idx", torch.tensor(tail_idx))
        self.num_classes = num_classes
        
        H = self.config.hidden_size
        self.roberta = RobertaModel(self.config, add_pooling_layer=False).from_pretrained(model_path, config=self.config)
        self.roberta.gradient_checkpointing_enable()

        H = self.config.hidden_size

        # ---- adapters (CXR처럼 3분기)
        self.adapt_h = BranchAdapter(H, r=256, dropout=0.0)
        self.adapt_t = BranchAdapter(H, r=256, dropout=0.0)
        self.adapt_b = BranchAdapter(H, r=256, dropout=0.0)

        # ---- env attention (Eq8)
        self.env_attn = AdditiveEnvAttention(dim=H, num_hiddens=H, dropout=0.1, attn_scale=0.1)
        # ---- losses
        # MFM (balanced logits에만)
        n_train = float(89098)
        self.loss = ASLwithClassWeight(cls_num_list, n_train)
      
        self.cls_bal  = BalancedCausalNormClassifier(num_classes, H)
        self.cls_head = HeadCausalNormClassifier(num_classes, H)
        self.cls_tail = TailCausalNormClassifier(num_classes, H) 

        self.env_attn = AdditiveEnvAttention(dim=H, num_hiddens=H, dropout=0.1, attn_scale=0.1)
        
        self.htb_loss = HeadTailBalancerLoss()
        self.htb_loss.PFM = self.loss
        self.mu = 0.9
        self.register_buffer("et_b", torch.zeros(H), persistent=True)
        self.register_buffer("et_h", torch.zeros(H), persistent=True)
        self.register_buffer("et_t", torch.zeros(H), persistent=True)
        
    def get_loss(self, z_b, z_h, z_t, z_h_hat, z_t_hat, targets):
        loss_main = self.loss(z_b, targets)
        loss_h = self.loss(z_h, targets)
        loss_t = self.loss(z_t, targets)
        loss_htb = self.htb_loss(z_h_hat, z_t_hat, z_b, targets)
        return loss_main + (loss_main + loss_h + loss_t) / 3.0 + loss_htb      
        

    def training_step(self, batch) -> dict[str, torch.Tensor]:
        data, targets, attention_mask = batch.data, batch.targets, batch.attention_mask
        out = self.forward(data, attention_mask, return_all=True)
        z_b, z_h, z_t, z_h_hat, z_t_hat = out["z_b"], out["z_h"], out["z_t"], out["z_h_nm"], out["z_t_nm"]
        loss = self.get_loss(z_b, z_h, z_t, z_h_hat, z_t_hat, targets)
        with torch.no_grad():
            self.et_b.mul_(self.mu).add_((1.0 - self.mu) * out["embed_mean_b"].detach().mean(0))
            self.et_h.mul_(self.mu).add_((1.0 - self.mu) * out["embed_mean_h"].detach().mean(0))
            self.et_t.mul_(self.mu).add_((1.0 - self.mu) * out["embed_mean_t"].detach().mean(0))
        probs = torch.sigmoid(z_b).detach()
        return {"logits": probs, "loss": loss, "targets": targets}


    def validation_step(self, batch) -> dict[str, torch.Tensor]:
        data, targets, attention_mask = batch.data, batch.targets, batch.attention_mask
        logits = self(data, attention_mask)
        loss = self.loss(logits, targets)
        logits = torch.sigmoid(logits)
        return {"logits": logits, "loss": loss, "targets": targets}

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        return_all=False
    ):
        r"""
        input_ids (torch.LongTensor of shape (batch_size, num_chunks, chunk_size))
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_labels)`, `optional`):
        """
                
        batch_size, num_chunks, chunk_size = input_ids.size()
        mask2d = attention_mask.view(-1, chunk_size) if attention_mask is not None else None
        mask1d = attention_mask.view(batch_size, -1)  if attention_mask is not None else None
        
        out = self.roberta(
            input_ids.view(-1, chunk_size),
            attention_mask=mask2d,
            return_dict=False,
        )
        hidden = out[0].view(batch_size, num_chunks * chunk_size, -1)  # [B, L, H]
        f0 = masked_mean_pool(hidden, mask1d)
        
        f_h = self.adapt_h(f0)
        f_t = self.adapt_t(f0)
        f_hat_b = self.adapt_b(f0)
 
        f_b = self.env_attn(f_hat_b, f_h, f_t)

        z_h, z_h_nm = self.cls_head(f_h, self.et_h)
        z_t, z_t_nm = self.cls_tail(f_t, self.et_t)
        z_b, z_b_nm = self.cls_bal(f_b, self.et_b)

        if not return_all:
            return z_b  # 기본 반환은 balanced logits로 유지

        return {
            "z_b": z_b, "z_h": z_h, "z_t": z_t,
            "z_b_nm": z_b_nm, "z_h_nm": z_h_nm, "z_t_nm": z_t_nm,
            "embed_mean_b": f_hat_b, "embed_mean_h": f_h, "embed_mean_t": f_t,
        }
    
