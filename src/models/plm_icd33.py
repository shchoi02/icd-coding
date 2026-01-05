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
import torch
import torch.utils.checkpoint
from torch import nn
from typing import Optional

from transformers import RobertaModel, AutoConfig

from src.models.modules.attention import LabelAttention

from src.losses.estimator import EstimatorCV
from src.losses.resample2 import ResampleLoss2



class PLMICD33(nn.Module):
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
        
        self.roberta = RobertaModel(
            self.config, add_pooling_layer=False
        ).from_pretrained(model_path, config=self.config)
        
        self.attention = LabelAttention(
            input_size=self.config.hidden_size,
            projection_size=self.config.hidden_size,
            num_classes=num_classes,
        )
        
        self.estimator = EstimatorCV(num_classes=num_classes) # 추가
            
        # self.loss = torch.nn.functional.binary_cross_entropy_with_logits
        self.slploss = ResampleLoss2(
            class_instance_nums=cls_num_list,
            use_slp=True,
            return_slp_debug=True,   # 필요 시 True
        )

    def get_loss(self, logits, targets):
        return self.loss(logits, targets)

    def training_step(self, batch) -> dict[str, torch.Tensor]:
        data, targets, attention_mask = batch.data, batch.targets, batch.attention_mask
        logits_full = self(data, attention_mask)
        
        active_idx = self.build_active_idx(targets, K_neg=2048)  # <- 메모리 맞춰 조절
        logits_ca = logits_full.index_select(1, active_idx)
        labels_ca = targets.index_select(1, active_idx)
        
        with torch.no_grad():
            prop_ca, cov_pos_ca, cov_neg_ca, sigma_ca, ro_ca, tao_ca = \
                self.estimator.update(labels_ca, logits_ca, active_idx=active_idx, device=logits_full.device)
        
        loss = self.slploss(
            norm_prop=prop_ca,
            nonzero_var_tensor=cov_pos_ca,
            zero_var_tensor=cov_neg_ca,
            normalized_sigma_cj=sigma_ca,
            normalized_ro_cj=ro_ca,
            normalized_tao_cj=tao_ca,
            cls_score=logits_ca,
            label=labels_ca,
        )
        
        logits = torch.sigmoid(logits_full)
        return {"logits": logits, "loss": loss, "targets": targets}


    def validation_step(self, batch) -> dict[str, torch.Tensor]:
        data, targets, attention_mask = batch.data, batch.targets, batch.attention_mask
        logits = self(data, attention_mask)
        loss = self.slploss.mfm(logits, targets.float())
        logits = torch.sigmoid(logits)
        return {"logits": logits, "loss": loss, "targets": targets}

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
    ):
        r"""
        input_ids (torch.LongTensor of shape (batch_size, num_chunks, chunk_size))
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_labels)`, `optional`):
        """

        batch_size, num_chunks, chunk_size = input_ids.size()
        outputs = self.roberta(
            input_ids.view(-1, chunk_size),
            attention_mask=attention_mask.view(-1, chunk_size)
            if attention_mask is not None
            else None,
            return_dict=False,
        )

        hidden_output = outputs[0].view(batch_size, num_chunks * chunk_size, -1)
        logits = self.attention(hidden_output)
        return logits
    
    @staticmethod
    @torch.no_grad()
    def build_active_idx(labels: torch.Tensor, K_neg: int = 512):
        """
        labels: [B, C_total] (0/1)
        return: active_idx [Ca] (pos union sampled neg)
        """
        dev = labels.device
        pos_mask = labels.any(dim=0)                      # [C_total]
        pos_idx = pos_mask.nonzero(as_tuple=False).squeeze(1)

        neg_pool = (~pos_mask).nonzero(as_tuple=False).squeeze(1)
        if neg_pool.numel() > 0 and K_neg > 0:
            k = min(K_neg, neg_pool.numel())
            perm = torch.randperm(neg_pool.numel(), device=dev)[:k]
            neg_idx = neg_pool[perm]
            active_idx = torch.unique(torch.cat([pos_idx, neg_idx], dim=0))
        else:
            active_idx = pos_idx

        return active_idx