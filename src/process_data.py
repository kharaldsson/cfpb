"""
TBD

"""

import logging
import os
import numpy as np
import pandas as pd

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

import torch

# from transformers import (
#     AutoModel
#     pipeline
# )

from datasets import (
    load_dataset,
    DatasetDict,
    Dataset,
    Features,
    ClassLabel,
    Value
)

from evaluate import load

from itertools import islice


log = logging.getLogger(__name__)

# Resolver to split a string x on a character y and return the (z-1)th element
OmegaConf.register_new_resolver("split", lambda x, y, z: x.split(y)[z])


@hydra.main(config_path="../config", config_name="process_data", version_base=None)
def process(cfg: DictConfig) -> None:
    """
    TBD

    """
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    if cfg.data.type == 'json':
        data = pd.read_json(cfg.data.path)
    elif cfg.data.type == 'csv':
        data = pd.read_csv(
            cfg.data.path,
            dtype={cfg.label_field: str, cfg.text_field: str},
            # nrows=1000 # TODO remove nrows
            ) 
    else:
        raise ValueError(f"Unspoorted data type: {cfg.data.dype}")
    
    log.info(f"Loaded data from {cfg.data.path}. Shape: {data.shape}")

    print(data.columns.names)
    data = data.dropna(subset=[cfg.text_field])
    data = data[data[cfg.text_field] != ""]

    log.info(f"Shape after dropping empty text_field: {data.shape}")

    print(cfg.label_field_allowed_values)
    data = data[data[cfg.label_field].isin(cfg.label_field_allowed_values) == True]
    log.info(f"Shape after filtering target values: {data.shape}")

    data = data[[cfg.text_field, cfg.label_field]].rename(
        columns={cfg.text_field: "text", cfg.label_field: "label"}
        ).copy()

    data = data.reset_index(drop=True)  # Drop the index to avoid __index_level_0__

    # Define Features
    features = Features({
        "text": Value("string"),
        "label": ClassLabel(names=list(data["label"].unique())) 
    })

    # Convert DataFrame to Dataset
    hf_dataset = Dataset.from_pandas(data, features=features)

    # Split dataset into train/valid/test
    split_percentages = cfg.splits 
    assert sum(split_percentages.values()) == 1.0, "Split percentages must sum to 1.0"

    # Shuffle and split
    hf_dataset = hf_dataset.shuffle(seed=cfg.random_state)
    split_dict = {
        "train": int(len(hf_dataset) * split_percentages["train"]),
        "valid": int(len(hf_dataset) * split_percentages["valid"])
    }
    split_dict["test"] = len(hf_dataset) - sum(split_dict.values())

    hf_dataset = hf_dataset.train_test_split(
        test_size=split_dict["test"] / len(hf_dataset), seed=cfg.random_state
    )
    valid_test = hf_dataset["test"].train_test_split(
        test_size=split_dict["valid"] / (split_dict["valid"] + split_dict["train"]), seed=cfg.random_state
    )
    hf_dataset["valid"] = valid_test["test"]
    hf_dataset["test"] = valid_test["train"]

    # Save the dataset
    output_dir = os.path.join(cfg.data.base_dir, cfg.dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    hf_dataset.save_to_disk(output_dir)
    log.info(f"Dataset saved to {output_dir}")

    # Log dataset splits
    for split in hf_dataset.keys():
        log.info(f"{split} split: {len(hf_dataset[split])} examples")


if __name__ == "__main__":
    process()