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


PATH = Path("./data/processed").expanduser()
DUMP_DIR = PATH.joinpath("ebnerd_predictions")
DUMP_DIR.mkdir(exist_ok=True, parents=True)
DT_NOW = dt.datetime.now()

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


embedding = 'xlm_roberta_base'
BATCH_SIZE = 32
learning_rate = 1e-4
HISTORY_SIZE =  35
MODEL_NAME = "NRMS-2025-01-13 15:54:33.945585"
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


df_test = pl.read_parquet(PATH.joinpath("test.parquet"))

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


test_dataloader = NRMSDataLoader(
    behaviors=df_test,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=True,
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

gc.collect()

print("loading model...")
model.model.load_weights(MODEL_WEIGHTS+"nrms.weights.h5")

pred_test = model.scorer.predict(test_dataloader)

df_test = add_prediction_scores(df_test, pred_test.tolist())

aucsc = AucScore()
print(len(df_test["labels"].to_list()))
print(df_test["scores"].to_list())
auc = aucsc.calculate(y_true=df_test["labels"].to_list(), y_pred=df_test["scores"].to_list())

print(f"Test AUC: {auc}")