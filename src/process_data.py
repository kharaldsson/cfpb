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
            nrows=1000000 # TODO remove nrows
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

    data = data.reset_index(drop=True)

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

    # Compute absolute sizes for splits
    total_len = len(hf_dataset)
    test_size = int(total_len * split_percentages["test"])
    valid_size = int(total_len * split_percentages["valid"])

    # Perform splits
    dataset_splits = hf_dataset.train_test_split(test_size=test_size, seed=cfg.random_state)
    train_valid_dataset = dataset_splits["train"].train_test_split(test_size=valid_size, seed=cfg.random_state)

    # Create a DatasetDict
    dataset_dict = DatasetDict({
        "train": train_valid_dataset["train"],
        "valid": train_valid_dataset["test"],
        "test": dataset_splits["test"]
    })

    log.info(f"Dataset: {dataset_dict}")

    # Save the dataset
    output_dir = os.path.join(cfg.data.base_dir, cfg.dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    dataset_dict.save_to_disk(output_dir)
    log.info(f"Dataset saved to {output_dir}")


if __name__ == "__main__":
    process()