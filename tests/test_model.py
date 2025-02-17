
from pathlib import Path
import tensorflow as tf
import datetime as dt
import polars as pl
import gc
import os
import numpy as np
import argparse

from tests import _PATH_DATA


from src.nrms_ml_ops.utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_USER_COL,
)

from src.nrms_ml_ops.utils._behaviors import (
    add_prediction_scores,
)
from src.nrms_ml_ops.evaluation import AucScore
from src.nrms_ml_ops.utils._articles import create_article_id_to_value_mapping

from src.nrms_ml_ops.dataloader import NRMSDataLoader
from src.nrms_ml_ops.model_config import hparams_nrms
from src.nrms_ml_ops.model import NRMSModel_docvec


os.environ["TOKENIZERS_PARALLELISM"] = "false"
tf.config.optimizer.set_jit(False)

weights_path = "./models/NRMS-2025-01-23 16:10:47.711802nrms.weights.h5"

def test_model_weights():
    MODEL_WEIGHTS = Path(weights_path).expanduser()  # Set the model weights path directly

    # Ensure the model directory exists
    if not MODEL_WEIGHTS.exists():
        raise FileNotFoundError(f"Error: The model path {MODEL_WEIGHTS} does not exist.")
        
        exit(1)
        
MODEL_WEIGHTS = Path(weights_path).expanduser()  # Set the model weights path directly

PATH = Path("./data/processed").expanduser()
DUMP_DIR = PATH.joinpath("ebnerd_predictions")
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


df_test = pl.read_parquet(os.path.join(_PATH_DATA,"test.parquet"))

df_articles = pl.read_parquet(os.path.join(_PATH_DATA,"articles.parquet"))

precomputed_embeddings = pl.read_parquet(os.path.join(_PATH_DATA, "xlm_roberta_base.parquet"))

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



def test_model_weights():
    """Test that the model weights are loaded correctly."""
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

    assert model.model.weights != None
    
    
def test_model_prediction():
    """Test that the model weights are loaded correctly."""
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
    
    df_test = pl.read_parquet(os.path.join(_PATH_DATA,"test.parquet"))

    pred_test = model.scorer.predict(test_dataloader)
    df_test = add_prediction_scores(df_test, pred_test.tolist())

    aucsc = AucScore()
    auc = aucsc.calculate(y_true=df_test["labels"].to_list(), y_pred=df_test["scores"].to_list())

    assert auc > 0.4

    print(f"Test AUC: {auc}") 

