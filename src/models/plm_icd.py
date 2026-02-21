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
from src.losses.dr import DRLoss
from src.losses.rlc import ReflectiveLabelCorrectorLoss
from src.losses.apl import APLLoss
from src.losses.ral import Ralloss
from src.losses.ce import CrossEntropyLoss


# class MaskedReduction(nn.Module):
#     def __init__(self,
#                  base_loss: nn.Module,
#                  class_mask: torch.Tensor | None = None,
#                  apply_mask_in_val: bool = False):
#         super().__init__()
#         self.base_loss = base_loss
#         self.apply_mask_in_val = apply_mask_in_val
#         if class_mask is not None:
#             self.register_buffer("class_mask", class_mask.to(torch.float32))
#         else:
#             self.class_mask = None

#     def forward(self, logits: torch.Tensor, targets: torch.Tensor, *, is_train: bool = True) -> torch.Tensor:
#         # per-element 손실로 뽑음
#         with torch.cuda.amp.autocast(enabled=False):
#             loss_elem = self.base_loss(logits.float(), targets.float(), reduction="none")  # (B, C)

#         # 학습 단계에서만 마스킹(옵션에 따라 검증에도 적용 가능)
#         if self.class_mask is not None and (is_train or self.apply_mask_in_val):
#             loss_elem = loss_elem * self.class_mask           # (B, C) * (C,)
#             denom = logits.shape[0] * torch.clamp(self.class_mask.sum(), min=1.0)
#             return loss_elem.sum() / denom

#         # 평가 단계: 원래 mean
#         return loss_elem.mean()

# class BCEWithLogits_Clamped(nn.Module):
#     """
#     순수 BCE-with-logits + 로짓 클램프만 적용
#     - ASL의 clip/focusing/가중 없음
#     - pos_weight 없음
#     - reduction: "none" | "sum" | "mean"
#     """
#     def __init__(self):
#         super().__init__()
#         self.eps = 1e-8

#     def forward(self, x: torch.Tensor, y: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
#         # assert torch.isfinite(x).all(), "non-finite logits"
#         # assert torch.isfinite(y).all(), "non-finite targets"
        
#         x_sigmoid = torch.sigmoid(x)   # torch.pow(sigmoid(x), 1)과 동일
#         xs_pos = x_sigmoid
#         xs_neg = 1 - x_sigmoid

#         # Basic CE with eps clamp (NaN/-inf 방지)
#         los_pos = y        * torch.log(xs_pos.clamp(min=self.eps))
#         los_neg = (1 - y)  * torch.log(xs_neg.clamp(min=self.eps))
#         loss = -(los_pos + los_neg)

#         # reduction
#         if reduction == "none":
#             return loss
#         elif reduction == "sum":
#             return loss.sum()
#         else:  # "mean"
#             return loss.mean()
    

class PLMICD(nn.Module):
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
        
        # self.loss = torch.nn.functional.binary_cross_entropy_with_logits
        
        # self.loss = Hill()
        
        # self.loss = FocalLoss()
        
        # self.loss = ResampleLoss( # CB Loss
        #     use_sigmoid=True,                 
        #     reweight_func='CB',                             
        #     class_freq=class_freq,
        #     neg_class_freq=neg_class_freq
        # )
                      
        self.loss = ResampleLoss(
            use_sigmoid    = True,
            class_freq     = class_freq,
            neg_class_freq = neg_class_freq,
            reweight_func  ='rebalance',
        )

        # self.loss = AsymmetricLoss()
        
        # self.loss = APLLoss()
        
        # self.loss = Ralloss()
        
        # self.loss = MultiGrainedFocalLoss()
        # self.loss.create_weight(cls_num_list)
        
        # self.loss = PriorFocalModifierLoss()
        # self.loss.create_co_occurrence_matrix(co_occurrence_matrix)
        # self.loss.create_weight(cls_num_list)       
        
        # self.loss = DRLoss()
    
        # self.loss = ReflectiveLabelCorrectorLoss(num_classes=num_classes, distribution=cls_num_list)

    # def get_loss(self, logits, targets, *, is_train: bool):
    #     return self.loss(logits, targets, is_train=is_train)

        
    def get_loss(self, logits, targets):
        return self.loss(logits, targets)

    def training_step(self, batch) -> dict[str, torch.Tensor]:
        data, targets, attention_mask = batch.data, batch.targets, batch.attention_mask
        logits = self(data, attention_mask)
        loss = self.get_loss(logits, targets)
        logits = torch.sigmoid(logits)       
        return {"logits": logits, "loss": loss, "targets": targets}


    def validation_step(self, batch) -> dict[str, torch.Tensor]:
        data, targets, attention_mask = batch.data, batch.targets, batch.attention_mask
        logits = self(data, attention_mask)
        loss = self.get_loss(logits, targets)
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
        # with torch.cuda.amp.autocast(enabled=False):
        outputs = self.roberta(
            input_ids.view(-1, chunk_size),
            attention_mask=attention_mask.view(-1, chunk_size)
            if attention_mask is not None
            else None,
            return_dict=False,
            )
        hidden_output = outputs[0].view(batch_size, num_chunks * chunk_size, -1)
        # assert torch.isfinite(hidden_output).all(), "NaN/Inf before LabelAttention"
        logits = self.attention(hidden_output)  
        
        # assert torch.isfinite(logits).all(), "NaN/Inf in logits"  
        return logits
    
        # # 0) 기본 shape
        # batch_size, num_chunks, chunk_size = input_ids.size()

        # # 1) Roberta 호출 "직전" 검증 (입력/마스크/길이)
        # ids2d  = input_ids.view(-1, chunk_size)
        # mask2d = attention_mask.view(-1, chunk_size) if attention_mask is not None else None

        # # (a) input_ids 유효범위/유한성
        # assert torch.isfinite(ids2d).all(), "NaN/Inf in input_ids (2D)"
        # assert (ids2d >= 0).all() and (ids2d < self.config.vocab_size).all(), \
        #     f"input_ids out of range [0,{self.config.vocab_size-1}]"

        # # (b) 길이 512 초과 여부
        # assert chunk_size <= 512, f"chunk_size {chunk_size} > 512 (max 512)"

        # # (c) attention_mask 타입/값/행전부패딩
        # if mask2d is not None:
        #     assert torch.isfinite(mask2d).all(), "NaN/Inf in attention_mask (2D)"
        #     assert mask2d.dtype in (torch.long, torch.int64, torch.int32, torch.uint8, torch.bool), \
        #         f"attention_mask dtype suspicious: {mask2d.dtype}"
        #     # 0/1 이외 값 있는지
        #     bad_val = ~((mask2d == 0) | (mask2d == 1))
        #     assert not bad_val.any(), f"attention_mask has non-(0/1) values at some positions"
        #     # 청크 단위 all-pad
        #     allpad_rows = (mask2d.sum(dim=1) == 0)
        #     assert not allpad_rows.any(), f"found all-pad chunks: {int(allpad_rows.sum())}"

        # 2) Roberta 호출 (fp32 권장: overflow 검사 명확)
        # with torch.cuda.amp.autocast(enabled=False):
        #     outputs = self.roberta(
        #         ids2d,
        #         attention_mask=mask2d if mask2d is not None else None,
        #         return_dict=False,
        #     )

        # # 3) Roberta 출력 유한성
        # last_hidden = outputs[0]  # (B*num_chunks, chunk_size, H)
        # assert torch.isfinite(last_hidden).all(), "NaN/Inf from Roberta last_hidden_state"

        # hidden_output = last_hidden.view(batch_size, num_chunks * chunk_size, -1)
        # assert torch.isfinite(hidden_output).all(), "NaN/Inf before LabelAttention (reshaped)"

        # # 4) Attention 및 로짓
        # logits = self.attention(hidden_output)
        # assert torch.isfinite(logits).all(), "NaN/Inf in logits (after LabelAttention)"

        # return logits
