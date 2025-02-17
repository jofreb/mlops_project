from pathlib import Path
import tensorflow as tf
import datetime as dt
import polars as pl
import gc
import os
import numpy as np


from utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_USER_COL,
)

from utils._behaviors import (
    add_prediction_scores,
)
from evaluation import AucScore
from utils._articles import create_article_id_to_value_mapping

from dataloader import NRMSDataLoader
from model_config import hparams_nrms
from model import NRMSModel_docvec
from google.cloud import storage


os.environ["TOKENIZERS_PARALLELISM"] = "false"
tf.config.optimizer.set_jit(False)


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")


# Define the model weights path
MODEL_WEIGHTS_LOCAL = Path("/tmp/nrms.weights.h5")

# Download the model weights from the cloud storage
download_blob("project_mlops_bucket", "models/nrms.weights.h5", str(MODEL_WEIGHTS_LOCAL))

MODEL_WEIGHTS = MODEL_WEIGHTS_LOCAL

# Ensure the model directory exists
if not MODEL_WEIGHTS.exists():
    print(f"Error: The model path {MODEL_WEIGHTS} does not exist.")
    exit(1)

cloud_path = "gs://project_mlops_bucket/data/processed/"
DUMP_DIR = Path(cloud_path + "ebnerd_predictions")
DUMP_DIR.mkdir(exist_ok=True, parents=True)
DT_NOW = dt.datetime.now()

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


embedding = "xlm_roberta_base"
BATCH_SIZE = 32
learning_rate = 1e-4
HISTORY_SIZE = 35
MODEL_NAME = MODEL_WEIGHTS.stem
LOG_DIR = DUMP_DIR.joinpath(f"./runs/{MODEL_NAME}")

COLUMNS = [
    DEFAULT_USER_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
]


df_test = pl.read_parquet(cloud_path + "test.parquet")

df_articles = pl.read_parquet(cloud_path + "articles.parquet")

precomputed_embeddings = pl.read_parquet(cloud_path + embedding + ".parquet")

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
model.model.load_weights(str(MODEL_WEIGHTS))

pred_test = model.scorer.predict(test_dataloader)
df_test = add_prediction_scores(df_test, pred_test.tolist())

aucsc = AucScore()
auc = aucsc.calculate(y_true=df_test["labels"].to_list(), y_pred=df_test["scores"].to_list())

print(f"Test AUC: {auc}")
pred_test = model.scorer.predict(test_dataloader)
df_test = add_prediction_scores(df_test, pred_test.tolist())

aucsc = AucScore()
auc = aucsc.calculate(y_true=df_test["labels"].to_list(), y_pred=df_test["scores"].to_list())

print(f"Test AUC: {auc}")
