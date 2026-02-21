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
from src.losses.focal import FocalLoss
from src.losses.hill import Hill
from src.losses.asl import AsymmetricLoss
from src.losses.mfm import MultiGrainedFocalLoss
from src.losses.pfm import PriorFocalModifierLoss
from src.losses.resample import ResampleLoss
from src.losses.rlc import ReflectiveLabelCorrectorLoss
from src.losses.htb import HeadTailBalancerLoss
from src.losses.mfm import MultiGrainedFocalLoss




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
        
        self.roberta_h = RobertaModel(self.config, add_pooling_layer=False).from_pretrained(model_path, config=self.config)
        self.roberta_b = RobertaModel(self.config, add_pooling_layer=False).from_pretrained(model_path, config=self.config)
        self.roberta_t = RobertaModel(self.config, add_pooling_layer=False).from_pretrained(model_path, config=self.config)

        
        self.att_head = LabelAttention(
            input_size=self.config.hidden_size,
            projection_size=self.config.hidden_size,
            # num_classes=num_classes
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
            # num_classes=num_classes
        )
        
        self.register_buffer("head_idx", torch.tensor(head_idx))
        self.register_buffer("tail_idx", torch.tensor(tail_idx))
        self.num_classes = num_classes

        self.rlc = ReflectiveLabelCorrectorLoss(num_classes=num_classes, distribution=cls_num_list)
        self.mfm = MultiGrainedFocalLoss()
        self.mfm.create_weight(cls_num_list) 
        self.htb = HeadTailBalancerLoss(PFM=self.mfm)
    
 
    def _composite_loss(self, head, tail, bal, labels):
        loss_r = self.rlc(bal, labels)
        loss_m = self.mfm(bal, labels)          
        loss_b = self.htb(head, tail, bal, labels) 
        return 0.2 * loss_r + self.lambda_m * loss_m + self.lambda_b * loss_b

    def get_loss(self, head, tail, bal, targets):
        return self._composite_loss(head, tail, bal, targets)

    def training_step(self, batch) -> dict[str, torch.Tensor]:
        data, targets, attention_mask = batch.data, batch.targets, batch.attention_mask
        z_head, z_tail, z_bal = self(data, attention_mask)
        loss = self.get_loss(z_head, z_tail, z_bal, targets)
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
    ):
        r"""
        input_ids (torch.LongTensor of shape (batch_size, num_chunks, chunk_size))
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_labels)`, `optional`):
        """

        batch_size, num_chunks, chunk_size = input_ids.size()
        flat_ids = input_ids.view(-1, chunk_size)
        flat_mask = attention_mask.view(-1, chunk_size) if attention_mask is not None else None

        out_h = self.roberta_h(flat_ids, attention_mask=flat_mask, return_dict=False)
        out_b = self.roberta_b(flat_ids, attention_mask=flat_mask, return_dict=False)
        out_t = self.roberta_t(flat_ids, attention_mask=flat_mask, return_dict=False)

        hidden_h = out_h[0].view(batch_size, num_chunks * chunk_size, -1)
        hidden_b = out_b[0].view(batch_size, num_chunks * chunk_size, -1)
        hidden_t = out_t[0].view(batch_size, num_chunks * chunk_size, -1)

        logits_head = self.att_head(hidden_h)
        logits_bal  = self.att_bal(hidden_b)
        logits_tail = self.att_tail(hidden_t)

        logits_head = self._scatter(logits_head, self.head_idx)
        logits_tail = self._scatter(logits_tail, self.tail_idx)

        return logits_head, logits_tail, logits_bal