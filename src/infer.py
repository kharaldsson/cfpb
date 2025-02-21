"""
Inference script using a locally saved model and the transformers pipeline.
"""

import os
import logging
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import pipeline, AutoTokenizer
from datasets import load_from_disk

from sklearn.metrics import classification_report


log = logging.getLogger(__name__)


def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def predict_batch(
        batch: dict, 
        pipeline, 
        pred_field_name: str,
        pred_desc_field_name: str,
        text_field_name: str,
        ) -> dict:
    """
    Performs text classification on a batch of text samples using the pipeline.

    Args:
        batch (dict): A batch of data containing text samples.
        pipeline (transformers.pipelines.Pipeline): The pipeline for inference.
        pred_field_name (str): The field name for storing predicted label indices.
        pred_desc_field_name (str): The field name for storing predicted label descriptions.
        text_field_name (str): The field name containing the input text samples.

    Returns:
        dict: The batch with added fields for the predicted label index and description.
    """
    # Get the list of text samples from the batch
    text_list = batch[text_field_name]

    # Perform inference using the pipeline
    results = pipeline(text_list, return_all_scores=True)

    # Extract predicted indices and descriptions
    predictions = [max(enumerate(item), key=lambda x: x[1]["score"])[0] for item in results]
    descriptions = [max(item, key=lambda x: x["score"])["label"] for item in results]

    # Add predictions to the batch
    batch[pred_field_name] = predictions
    batch[pred_desc_field_name] = descriptions

    return batch



@hydra.main(config_path="../config", config_name="infer", version_base=None)
def infer(cfg: DictConfig) -> None:
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    device = get_device()
    log.info(f"Using device: {device}")

    # Load the test dataset from disk
    ds = hydra.utils.instantiate(cfg.dataset, _convert_="object")
    log.info("Dataset loaded.")
    test_split = cfg.get("test_split", "test")
    if test_split not in ds:
        raise ValueError(f"Test split '{test_split}' not found in dataset.")
    test_ds = ds[test_split]
    log.info(f"Loaded test split '{test_split}' with {len(test_ds)} samples.")

    # # Instantiate the model
    # # Get label info
    # labels, label2id, id2label = get_class_indices(dataset=ds, label_field=cfg.label_field)

    # Instantiate the model
    model: AutoModel = hydra.utils.instantiate(
        cfg.model, 
        # num_labels=len(labels),
        # id2label=id2label,
        # label2id=label2id,
        _convert_="object"
        )
    log.info(f"Loaded model from {cfg.saved_model_path}")

    model.to(device)

    # Instantiate the tokenizer
    tokenizer: AutoTokenizer = hydra.utils.instantiate(cfg.tokenizer, _convert_="object")
    log.info("Processor instantiated.")

    classifier = hydra.utils.instantiate(
        cfg.pipeline,
        model=model,
        tokenizer=tokenizer,
        device=device,
        _convert_="object",
    )
    log.info(f"Pipeline instantiated.")


    # Run inference and collect predictions
    batch_size = cfg.get("batch_size", 8)
    # Run inference on the dataset using map with batched processing
    test_ds = test_ds.map(
        lambda batch: predict_batch(
            batch=batch, 
            pipeline=classifier,
            pred_field_name=cfg.prediction_field,
            pred_desc_field_name=cfg.prediction_desc_field, 
            text_field_name=cfg.text_field,
            ),
        batched=True,
        batch_size=batch_size,
    )
    log.info("Inference completed.")

    # Save pred to file
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"output_dir={output_dir}")
    output_file = os.path.join(output_dir, cfg.get("predictions_fname", "predictions"))
    test_ds.save_to_disk(output_file)

        # Evaluation component
    if cfg.get("label_field") in test_ds.column_names:
        ground_truth_field = cfg.label_field
        prediction_field = cfg.prediction_field

        # Extract ground truth and predictions
        y_true = test_ds[ground_truth_field]
        y_pred = test_ds[prediction_field]

        # Generate classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0.0)
        log.info("Evaluation Metrics:")
        log.info(f"\n {classification_report(y_true, y_pred, zero_division=0.0)}")

        # Save evaluation metrics to a file
        metrics_file = os.path.join(output_dir, "classification_report.json")
        with open(metrics_file, "w") as f:
            import json
            json.dump(report, f, indent=4)
        log.info(f"Classification report saved to {metrics_file}")


if __name__ == "__main__":
    infer()