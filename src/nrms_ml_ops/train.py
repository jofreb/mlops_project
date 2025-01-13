from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import tensorflow as tf
import datetime as dt
import polars as pl
import gc
import os
from pathlib import Path
import sys
import numpy as np
import yaml


from utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_SUBTITLE_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_TITLE_COL,
    DEFAULT_USER_COL,
    DEFAULT_IMPRESSION_TIMESTAMP_COL,
)

from utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    add_known_user_column,
    add_prediction_scores,
    truncate_history,
)
from evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore
from utils._articles import convert_text2encoding_with_transformers
from utils._polars import concat_str_columns, slice_join_dataframes
from utils._articles import create_article_id_to_value_mapping
from utils._nlp import get_transformers_word_embeddings
from utils._python import write_submission_file, rank_predictions_by_score

from dataloader import NRMSDataLoader
from model_config import hparams_nrms
from model import NRMSModel_docvec

from typing import List, Dict, Any, Tuple, Optional, Union
from datetime import datetime, timedelta


os.environ["TOKENIZERS_PARALLELISM"] = "false"
tf.config.optimizer.set_jit(False)


def ebnerd_from_path(path: Path, history_size: int = 30) -> pl.DataFrame:
    """
    Load ebnerd - function
    """
    df_history = (
        pl.scan_parquet(path.joinpath("history.parquet"))
        .select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL)
        .pipe(
            truncate_history,
            column=DEFAULT_HISTORY_ARTICLE_ID_COL,
            history_size=history_size,
            padding_value=0,
            enable_warning=False,
        )
    )
    df_behaviors = (
        pl.scan_parquet(path.joinpath("behaviors.parquet"))
        .collect()
        .pipe(
            slice_join_dataframes,
            df2=df_history.collect(),
            on=DEFAULT_USER_COL,
            how="left",
        )
    )
    return df_behaviors


PATH = Path("./data/processed").expanduser()
DUMP_DIR = PATH.joinpath("ebnerd_predictions")
DUMP_DIR.mkdir(exist_ok=True, parents=True)
DT_NOW = dt.datetime.now()

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


EPOCHS = 10
embedding = 'xlm_roberta_base'
BATCH_SIZE = 32
learning_rate = 1e-4
HISTORY_SIZE =  35

MODEL_NAME = f"NRMS-{DT_NOW}"

# Create the folder, including any intermediate directories

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

df_train = (
    ebnerd_from_path(PATH.joinpath("train"), history_size=HISTORY_SIZE)
    .sample(fraction=1)
    .select(COLUMNS)
    .pipe(
        sampling_strategy_wu2019,
        npratio=4,
        shuffle=True,
        with_replacement=True,
        seed=123,
    )
    .pipe(create_binary_labels_column)
)


df_validation= (
    ebnerd_from_path(PATH.joinpath("validation"), history_size=HISTORY_SIZE)
    .sample(fraction=1)
    .select(COLUMNS)
    .pipe(
        sampling_strategy_wu2019,
        npratio=4,
        shuffle=True,
        with_replacement=True,
        seed=123,
    )
    .pipe(create_binary_labels_column)
)


df_test = (
    ebnerd_from_path(PATH.joinpath("test"), history_size=HISTORY_SIZE)
    .with_columns(
        pl.col(DEFAULT_INVIEW_ARTICLES_COL)
        .list.first()
        .alias(DEFAULT_CLICKED_ARTICLES_COL)
    )
    .select(COLUMNS)
    .with_columns(
        pl.col(DEFAULT_INVIEW_ARTICLES_COL)
        .list.eval(pl.element() * 0)
        .alias(DEFAULT_LABELS_COL)
    )
)


df_articles = pl.read_parquet(PATH.joinpath("articles.parquet"))

precomputed_embeddings = pl.read_parquet(PATH.joinpath(embedding+".parquet"))

precomputed_embeddings = precomputed_embeddings.filter(precomputed_embeddings['article_id'].is_in(df_articles['article_id']))
precomputed_embeddings = precomputed_embeddings.rename({'FacebookAI/xlm-roberta-base': 'embedding'})

pre_embs = np.array([precomputed_embeddings['embedding'][0]])

article_mapping = create_article_id_to_value_mapping(
    df=precomputed_embeddings,
    value_col="embedding",  # Column containing precomputed embeddings
    article_col="article_id",  # Column containing article IDs
)
 

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
    monitor="val_loss",
    mode="max",
    patience=3,
    restore_best_weights=True,
)

# ModelCheckpoint:
modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=MODEL_WEIGHTS+"nrms.weights.h5",
    monitor="val_loss",
    mode="max",
    save_best_only=False,
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

hist = model.model.fit(
    train_dataloader,
    validation_data=val_dataloader,
    epochs=EPOCHS,
    callbacks=[tensorboard_callback, early_stopping, modelcheckpoint, lr_scheduler],
)

tr_loss = hist.history['loss']
val_loss = hist.history['val_loss']
losses = [tr_loss, val_loss]

gc.collect()

print("saving model...")
model.model.save_weights(MODEL_WEIGHTS+"nrms.weights.h5")


print("Correctly ended")