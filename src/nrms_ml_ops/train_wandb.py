from pathlib import Path
import tensorflow as tf
import datetime as dt
import polars as pl
import gc
import os
import sys
import numpy as np
from loguru import logger

from utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_USER_COL,
)

import argparse

from utils._articles import create_article_id_to_value_mapping

from dataloader import NRMSDataLoader
from model_config import hparams_nrms
from model import NRMSModel_docvec


import wandb
from wandb.integration.keras import WandbCallback


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with specified hyperparameters")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    return parser.parse_args()


os.environ["TOKENIZERS_PARALLELISM"] = "false"
tf.config.optimizer.set_jit(False)

args = parse_args()
PATH = Path("./data/processed").expanduser()
DUMP_DIR = PATH.joinpath("ebnerd_predictions")
DUMP_DIR.mkdir(exist_ok=True, parents=True)
DT_NOW = dt.datetime.now()

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


EPOCHS = args.epochs
embedding = "xlm_roberta_base"
BATCH_SIZE = args.batch_size
learning_rate = args.learning_rate
HISTORY_SIZE = 35

MODEL_NAME = f"NRMS-{DT_NOW}"

# Create the folder, including any intermediate directories
wandb.init(
    project="nrms_mlops",
    config={"lr": learning_rate, "batch_size": BATCH_SIZE, "epochs": EPOCHS},
    sync_tensorboard=True,
)

MODEL_WEIGHTS = f"./models/{MODEL_NAME}"
Path(MODEL_WEIGHTS).mkdir(parents=True, exist_ok=True)
LOG_DIR = DUMP_DIR.joinpath(f"./runs/{MODEL_NAME}")

COLUMNS = [
    DEFAULT_USER_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
]

df_train = pl.read_parquet(PATH.joinpath("train.parquet"))


df_validation = pl.read_parquet(PATH.joinpath("validation.parquet"))
logger.remove()  # Remove the default logger
logger.add(sys.stdout, level="DEBUG")  # Add a new logger with WARNING level
logger.debug("Data loaded")


df_articles = pl.read_parquet(PATH.joinpath("articles.parquet"))

precomputed_embeddings = pl.read_parquet(PATH.joinpath(embedding + ".parquet"))

precomputed_embeddings = precomputed_embeddings.filter(
    precomputed_embeddings["article_id"].is_in(df_articles["article_id"])
)
precomputed_embeddings = precomputed_embeddings.rename({"FacebookAI/xlm-roberta-base": "embedding"})

pre_embs = np.array([precomputed_embeddings["embedding"][0]])

article_mapping = create_article_id_to_value_mapping(
    df=precomputed_embeddings,
    value_col="embedding",  # Column containing precomputed embeddings
    article_col="article_id",  # Column containing article IDs
)
logger.debug("Article Data created")

train_dataloader = NRMSDataLoader(
    behaviors=df_train,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=False,
    batch_size=BATCH_SIZE,
)
val_dataloader = NRMSDataLoader(
    behaviors=df_validation,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=False,
    batch_size=BATCH_SIZE,
)

hparams_nrms.history_size = HISTORY_SIZE
hparams_nrms.learning_rate = learning_rate
hparams_nrms.title_size = pre_embs[0].shape[0]


model = NRMSModel_docvec(
    hparams=hparams_nrms,
    seed=42,
)

model.model.compile(
    optimizer=model.model.optimizer,
    loss=model.model.loss,
    metrics=["AUC"],
)

# CALLBACKS
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)

# Earlystopping:
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=3,
    restore_best_weights=True,
)

# ModelCheckpoint:
modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=MODEL_WEIGHTS + "nrms.weights.h5",
    monitor="val_loss",
    mode="max",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
)

# Learning rate scheduler:
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    mode="max",
    factor=0.2,
    patience=2,
    min_lr=learning_rate,
)

# Wandb Callback
wandb_callback = WandbCallback(
    monitor="val_auc",
    verbose=0,
    mode="auto",
    save_weights_only=(False),
    log_weights=(False),
    log_gradients=(False),
    save_model=(True),
    training_data=None,
    validation_data=None,
    labels=None,
    predictions=36,
    generator=None,
    input_type=None,
    output_type=None,
    log_evaluation=(False),
    validation_steps=None,
    class_colors=None,
    log_batch_frequency=None,
    log_best_prefix="best_",
    save_graph=(True),
    validation_indexes=None,
    validation_row_processor=None,
    prediction_row_processor=None,
    infer_missing_processors=(True),
    log_evaluation_frequency=0,
    compute_flops=(False),
)


hist = model.model.fit(
    train_dataloader,
    validation_data=val_dataloader,
    epochs=EPOCHS,
    callbacks=[tensorboard_callback, early_stopping, modelcheckpoint, lr_scheduler, wandb_callback],
)
logger.debug("Training finished")

val_auc = hist.history["val_auc"][-1]
val_loss = hist.history["val_loss"][-1]
tr_auc = hist.history["auc"][-1]
tr_loss = hist.history["loss"][-1]


gc.collect()


# logger.debug("Saving model")
model.model.save_weights(MODEL_WEIGHTS + "nrms.weights.h5")

# artifact = wandb.Artifact(
#         name="nrms_model",
#         type="model",
#         description="A model trained to predict which articles a user will click on",
#         metadata={"validation_AUC": val_auc, "training_AUC": tr_auc, "validation_loss": val_loss, "training_loss": tr_loss},
#     )
# artifact.add_file(MODEL_WEIGHTS+"nrms.weights.h5")
# wandb.log_artifact(artifact)

# model.model.save(MODEL_WEIGHTS + "nrms_model", save_format="tf")  # Saves in SavedModel format
# artifact = wandb.Artifact(
#     name="nrms_model",
#     type="model",
#     description="A model trained to predict which articles a user will click on for model registry",
#     metadata={
#         "validation_AUC": val_auc,
#         "training_AUC": tr_auc,
#         "validation_loss": val_loss,
#         "training_loss": tr_loss,
#     },
# )
# artifact.add_dir(MODEL_WEIGHTS + "nrms_model")
# wandb.log_artifact(artifact)


weights_artifact = wandb.Artifact(
    name="nrms_model_weights",
    type="model-weights",
    description="NRMS model weights extracted from SavedModel.",
)
weights_artifact.add_file(MODEL_WEIGHTS + "nrms.weights.h5")
wandb.log_artifact(weights_artifact)
# run.log_artifact(artifact)

# token wandb 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

logger.debug("Correctly ended")
