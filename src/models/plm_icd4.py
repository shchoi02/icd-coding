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

from src.models.modules.attention import LabelAttentionBalanced, LabelAttentionHead, LabelAttentionTail, BalancedCausalNormVec
import torch.nn.functional as F
from src.losses.mfm import MultiGrainedFocalLoss
from src.losses.rlc import ReflectiveLabelCorrectorLoss

    
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


class PLMICD4(nn.Module):
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

        self.attn_h = LabelAttentionHead(input_size=H, projection_size=H, num_classes=num_classes)
        self.attn_t = LabelAttentionTail(input_size=H, projection_size=H, num_classes=num_classes)
        self.attn_b = LabelAttentionBalanced(input_size=H, projection_size=H, num_classes=num_classes)
      
        self.env_attn = AdditiveEnvAttention(dim=H, num_hiddens=H, dropout=0.1, attn_scale=0.1)
        self.cls_bal_vec = BalancedCausalNormVec(num_classes, H, use_effect=False)
       
        # MFM (balanced logits에만)
        self.rlc = ReflectiveLabelCorrectorLoss(num_classes=num_classes, distribution=cls_num_list)
        self.loss = MultiGrainedFocalLoss()
        self.loss.create_weight(cls_num_list) 
        self.htb_loss = HeadTailBalancerLoss(PFM=self.loss)
        self.mu = 0.9
        self.register_buffer("et_b", torch.zeros(H), persistent=True)
        self.register_buffer("et_h", torch.zeros(H), persistent=True)
        self.register_buffer("et_t", torch.zeros(H), persistent=True)
        
    def get_loss(self, z_b, z_h, z_t, z_h_hat, z_t_hat, targets):
        loss_rlc = self.rlc(z_b, targets)
        loss_main = self.loss(z_b, targets)
        loss_htb = self.htb_loss(z_h_hat, z_t_hat, z_b, targets)
        return 0.2 * loss_rlc + loss_main + loss_htb      

    def training_step(self, batch) -> dict[str, torch.Tensor]:
        data, targets, attention_mask = batch.data, batch.targets, batch.attention_mask
        out = self.forward(data, attention_mask, return_all=True)
        z_b, z_h, z_t, z_h_hat, z_t_hat = out["z_b"], out["z_h"], out["z_t"], out["z_h_nm"], out["z_t_nm"]
        loss = self.get_loss(z_b, z_h, z_t, z_h_hat, z_t_hat, targets)
        with torch.no_grad():
            self.et_b.mul_(self.mu).add_((1.0 - self.mu) * out["feat_b"].detach().mean(0))
            self.et_h.mul_(self.mu).add_((1.0 - self.mu) * out["feat_h"].detach().mean(0))
            self.et_t.mul_(self.mu).add_((1.0 - self.mu) * out["feat_t"].detach().mean(0))
        probs = torch.sigmoid(z_b).detach()
        return {"logits": probs, "loss": loss, "targets": targets}


    def validation_step(self, batch):
        data, targets, attention_mask = batch.data, batch.targets, batch.attention_mask
        out = self.forward(data, attention_mask, return_all=True)
        z_b, z_h, z_t = out["z_b"], out["z_h"], out["z_t"]
        z_h_hat, z_t_hat = out["z_h_nm"], out["z_t_nm"]

        loss = self.get_loss(z_b, z_h, z_t, z_h_hat, z_t_hat, targets)
        logits = torch.sigmoid(z_b)
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
        
        # head/tail/bal logits + embed_mean 받기
        z_h, z_h_nm, f_h = self.attn_h(hidden, attention_mask_1d=mask1d, embed=self.et_h)
        z_t, z_t_nm, f_t = self.attn_t(hidden, attention_mask_1d=mask1d, embed=self.et_t)
        z_b, z_b_nm, f_hat_b = self.attn_b(hidden, attention_mask_1d=mask1d, embed=None)
        
        f_b = self.env_attn(f_hat_b, f_h, f_t)
        z_b, z_b_nm = self.cls_bal_vec(f_b, embed=None)  # [B,C]

        if not return_all:
            return z_b

        return {
            "z_b": z_b, "z_h": z_h, "z_t": z_t,
            "z_b_nm": z_b_nm, "z_h_nm": z_h_nm, "z_t_nm": z_t_nm,
            "feat_b": f_b, "feat_h": f_h, "feat_t": f_t,
        }

    