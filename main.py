import logging
import math
import os
import json
import numpy as np
from pathlib import Path
from contextlib import contextmanager

import hydra
import torch
from omegaconf import OmegaConf
from rich.pretty import pprint
import torch, subprocess, re, logging

from src.data.data_pipeline import data_pipeline
from src.factories import (
    get_callbacks,
    get_dataloaders,
    get_datasets,
    get_lookups,
    get_lr_scheduler,
    get_metric_collections,
    get_model,
    get_optimizer,
    get_text_encoder,
    get_transform,
)
from src.trainer.trainer import Trainer
from src.utils.seed import set_seed

LOGGER = logging.getLogger(name=__file__)
LOGGER.setLevel(logging.INFO)


def _raw_targets_iter(samples, target_col: str = "target"):
    for s in samples:
        if isinstance(s, dict):
            yield s.get("targets", s.get(target_col))
        else:               
            yield s[1]


def get_cls_num_list(
    data,
    label_transform,
    split: str = "train",
) -> list[int]:
    samples = getattr(data, split)      
    C = label_transform.num_classes
    counts = torch.zeros(C, dtype=torch.long)

    for raw in _raw_targets_iter(samples):
        idxs = label_transform.get_indices(raw)
        counts[idxs] += 1

    return counts.tolist()


def build_co_occurrence_matrix(
    data,
    label_transform,
    split: str = "train",
    *,
    device: str | torch.device = "cuda",
) -> np.ndarray:

    samples = getattr(data, split)
    C      = label_transform.num_classes
    dev    = torch.device(device)
    counts = torch.zeros((C, C), dtype=torch.float32, device=dev)
    with torch.no_grad():
        for raw in _raw_targets_iter(samples):
            idxs = label_transform.get_indices(raw).to(dev) 
            if idxs.numel():                                  
                counts[idxs[:, None], idxs] += 1.0
    counts.fill_diagonal_(0)
    return counts.cpu().numpy().astype(np.float32)


def compute_class_stats(
    data,
    label_transform,
    split: str = "train",
    *,
    device: str | torch.device = "cpu",
):
    samples = getattr(data, split)
    C   = label_transform.num_classes
    dev = torch.device(device)

    pos_freq = torch.zeros(C, dtype=torch.long, device=dev)
    for raw in _raw_targets_iter(samples):
        idxs = label_transform.get_indices(raw).to(dev)
        pos_freq[idxs] += 1

    total_samples = len(samples)
    neg_freq = torch.full_like(pos_freq, total_samples) - pos_freq
    return pos_freq, neg_freq


def get_head_tail_indices(label_transform):
    with open('/home/mixlab/tabular/icd-coding/files/data/mimiciv_icd10/icd10_longtail_split.json', "r", encoding="utf-8") as jf:
        ht_dict = json.load(jf)

    head_codes = list(ht_dict["head"].keys())
    tail_codes = list(ht_dict["tail"].keys())
    medium_codes = list(ht_dict["medium"].keys())

    head_idx = label_transform.get_indices(head_codes)
    tail_idx = label_transform.get_indices(tail_codes)
    medium_idx = label_transform.get_indices(medium_codes)

    return head_idx, tail_idx, medium_idx


def deterministic() -> None:
    """Run experiment deterministically. There will still be some randomness in the backward pass of the model."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    import torch

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: OmegaConf) -> None:
    if cfg.deterministic:
        deterministic()
    else:
        import torch

    set_seed(cfg.seed)

    # Check if CUDA_VISIBLE_DEVICES is set
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        if cfg.gpu != -1 and cfg.gpu is not None and cfg.gpu != "":
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                ",".join([str(gpu) for gpu in cfg.gpu])
                if isinstance(cfg.gpu, list)
                else str(cfg.gpu)
            )

        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pprint(f"Device: {device}")
    pprint(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

    data = data_pipeline(config=cfg.data)

    text_encoder = get_text_encoder(
        config=cfg.text_encoder, data_dir=cfg.data.dir, texts=data.get_train_documents
    )
    label_transform = get_transform(
        config=cfg.label_transform,
        targets=data.all_targets,
        load_transform_path=cfg.load_model,
    )
    text_transform = get_transform(
        config=cfg.text_transform,
        texts=data.get_train_documents,
        text_encoder=text_encoder,
        load_transform_path=cfg.load_model,
    )
    data.truncate_text(cfg.data.max_length)
    data.transform_text(text_transform.batch_transform)

    lookups = get_lookups(
        config=cfg.lookup,
        data=data,
        label_transform=label_transform,
        text_transform=text_transform,
    )

    cls_num_list = get_cls_num_list(data, label_transform)
    # file_path = 'cls_num_list.txt'
    # with open(file_path, 'w') as file:
    #     for item in cls_num_list:
    #         file.write(str(item) + '\n')
    
    co_occurrence_matrix = build_co_occurrence_matrix(data, label_transform)
    # np.save("co_occurrence_matrix.npy", co_occurrence_matrix)
    
    class_freq, neg_class_freq = compute_class_stats(data, label_transform)
    
    head_idx, tail_idx, medium_idx = get_head_tail_indices(label_transform)
        

    model = get_model(
        config=cfg.model, data_info=lookups.data_info, text_encoder=text_encoder, 
        cls_num_list=cls_num_list, 
        head_idx=head_idx, tail_idx=tail_idx,
        co_occurrence_matrix=co_occurrence_matrix, 
        class_freq=class_freq, neg_class_freq=neg_class_freq
    )
    model.to(device)

    # print data info
    pprint(lookups.data_info)

    metric_collections = get_metric_collections(
        config=cfg.metrics,
        number_of_classes=lookups.data_info["num_classes"],
        code_system2code_indices=lookups.code_system2code_indices,
        split2code_indices=lookups.split2code_indices,
    ) 
    
    datasets = get_datasets(
        config=cfg.dataset,
        data=data,
        text_transform=text_transform,
        label_transform=label_transform,
        lookups=lookups,
    )

    dataloaders = get_dataloaders(config=cfg.dataloader, datasets_dict=datasets)
    
    optimizer = get_optimizer(config=cfg.optimizer, model=model)
    accumulate_grad_batches = int(
        max(cfg.dataloader.batch_size / cfg.dataloader.max_batch_size, 1)
    )
    num_training_steps = (
        math.ceil(len(dataloaders["train"]) / accumulate_grad_batches)
        * cfg.trainer.epochs
    )

    lr_scheduler = get_lr_scheduler(
        config=cfg.lr_scheduler,
        optimizer=optimizer,
        num_training_steps=num_training_steps,
    )
    callbacks = get_callbacks(config=cfg.callbacks)
    
    trainer = Trainer(
        config=cfg,
        data=data,
        model=model,
        optimizer=optimizer,
        dataloaders=dataloaders,
        metric_collections=metric_collections,
        callbacks=callbacks,
        lr_scheduler=lr_scheduler,
        lookups=lookups,
        accumulate_grad_batches=accumulate_grad_batches,
        label_transform=label_transform
    ).to(device)

    if cfg.load_model:
        trainer.experiment_path = Path(cfg.load_model)
    
    trainer.fit()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
