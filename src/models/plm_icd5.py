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
from src.losses.asl import ASLwithClassWeight
from src.losses.htb import HeadTailBalancerLoss


class PLMICD5(nn.Module):
    def __init__(self, num_classes: int, model_path: str,
                 cls_num_list, 
                 head_idx = None, tail_idx = None,
                 co_occurrence_matrix = None,
                 class_freq = None, neg_class_freq = None,
                 **kwargs):
        super().__init__()
        
        self.lambda_r = 0.2
        self.lambda_m = 1.0
        self.lambda_b = 1.0
        
        self.config = AutoConfig.from_pretrained(
            model_path, num_labels=num_classes, finetuning_task=None
        )
        
        self.roberta = RobertaModel(
            self.config, add_pooling_layer=False
        ).from_pretrained(model_path, config=self.config)
        
        self.att_head = LabelAttention(
            input_size=self.config.hidden_size,
            projection_size=self.config.hidden_size,
            num_classes=len(head_idx),
        )
        self.att_bal = LabelAttention(
            input_size=self.config.hidden_size,
            projection_size=self.config.hidden_size,
            num_classes=num_classes,
        )
        self.att_tail = LabelAttention(
            input_size=self.config.hidden_size,
            projection_size=self.config.hidden_size,
            num_classes=len(tail_idx),
        )
        
        self.register_buffer("head_idx", torch.tensor(head_idx))
        self.register_buffer("tail_idx", torch.tensor(tail_idx))
        self.num_classes = num_classes
        
        if not torch.is_tensor(cls_num_list):
            cls_num_list = torch.tensor(cls_num_list, dtype=torch.float32)
        cls_num_list = cls_num_list.float()

        head_counts = cls_num_list.index_select(0, self.head_idx)
        tail_counts = cls_num_list.index_select(0, self.tail_idx)
        n_train = float(89098) # 110441
        self.wasl_b = ASLwithClassWeight(cls_num_list, n_train)  # (C,)
        self.wasl_h = ASLwithClassWeight(head_counts, n_train)   # (|H|)
        self.wasl_t = ASLwithClassWeight(tail_counts, n_train)   # (|T|)
        self.htb = HeadTailBalancerLoss(PFM=self.wasl_b)
 
    def get_loss(self, head, tail, bal, z_h_part, z_t_part, labels):
        y_h = labels.index_select(1, self.head_idx)  # (B, |H|)
        y_t = labels.index_select(1, self.tail_idx)  # (B, |T|)
        loss_m = self.wasl_b(bal, labels) 
        loss_wasl = (loss_m + self.wasl_h(z_h_part, y_h) + self.wasl_t(z_t_part, y_t)) / 3.0
        loss_b = self.htb(head, tail, bal, labels) 
        return loss_m + loss_wasl + loss_b

    def training_step(self, batch) -> dict[str, torch.Tensor]:
        data, targets, attention_mask = batch.data, batch.targets, batch.attention_mask
        z_head, z_tail, z_bal, z_h_part, z_t_part = self(data, attention_mask, return_all=True)
        loss = self.get_loss(z_head, z_tail, z_bal, z_h_part, z_t_part, targets)
        logits = torch.sigmoid(z_bal)
        return {"logits": logits, "loss": loss, "targets": targets}

    def validation_step(self, batch) -> dict[str, torch.Tensor]:
        data, targets, attention_mask = batch.data, batch.targets, batch.attention_mask
        z_head, z_tail, z_bal = self(data, attention_mask)
        loss = self.get_loss(z_head, z_tail, z_bal, targets)
        logits = torch.sigmoid(z_bal)
        return {"logits": logits, "loss": loss, "targets": targets}
     
    def _scatter(self, part_logits: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        B = part_logits.size(0)
        full = part_logits.new_zeros(B, self.num_classes)
        full.index_copy_(1, idx, part_logits) 
        return full

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        return_all=False,
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
        
        logits_head_part = self.att_head(hidden_output)
        logits_bal  = self.att_bal(hidden_output) 
        logits_tail_part = self.att_tail(hidden_output)
        
        logits_head = self._scatter(logits_head_part, self.head_idx)
        logits_tail = self._scatter(logits_tail_part, self.tail_idx) 
        
        
        if not return_all:
            return logits_bal # 기본 반환은 balanced logits로 유지
        
        return logits_head, logits_tail, logits_bal, logits_head_part, logits_tail_part
