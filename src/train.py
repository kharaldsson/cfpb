"""
TBD

"""

import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

import torch


from datasets import (
    load_dataset,
    DatasetDict,
    Dataset,
)

from evaluate import load

from itertools import islice


log = logging.getLogger(__name__)

# Resolver to split a string x on a character y and return the (z-1)th element
OmegaConf.register_new_resolver("split", lambda x, y, z: x.split(y)[z])


def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_class_indices(dataset, label_field):
    classes = [class_ for class_ in dataset['train'].features[label_field].names if class_]
    class2id = {class_:id for id, class_ in enumerate(classes)}
    id2class = {id:class_ for class_, id in class2id.items()}
    return classes, class2id, id2class


@hydra.main(config_path="../config", config_name="train", version_base=None)
def train(cfg: DictConfig) -> None:
    """
    TBD

    """
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    device = get_device()
    log.info(f"Device: {device}")

    # Load the dataset
    ds = hydra.utils.instantiate(cfg.dataset, _convert_="object")
    log.info("Dataset loaded.")

    # Get label info
    labels, label2id, id2label = get_class_indices(dataset=ds, label_field=cfg.label_field)

    # Instantiate the model
    model: AutoModelForSequenceClassification = hydra.utils.instantiate(
        cfg.model, 
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        _convert_="object"
        )
    log.info("Model loaded.")

    model.to(device)

    # Instantiate the tokenizer
    tokenizer: AutoTokenizer = hydra.utils.instantiate(cfg.tokenizer, _convert_="object")

    # Set up data collator
    data_collator = hydra.utils.instantiate(cfg.collator, tokenizer=tokenizer,  return_tensors="pt", _convert_="object")


    if not cfg.data.get("is_tokenized", False):
        log.info("Tokenizing dataset.")
        ds = ds.map(
            lambda examples: tokenizer(
                examples[cfg.text_field],
                truncation=True,
                padding='max_length',
                max_length=cfg.tokenizer.model_length,
            ),
            batched=True,
        )
    else:
        log.info("Dataset is already tokenized; proceeding.")


    # Instantiate trainer
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        model=model,
        train_dataset=ds[cfg.train_split],
        eval_dataset=ds[cfg.eval_split],
        tokenizer=tokenizer,
        data_collator=data_collator,
        _convert_="object",
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    train()