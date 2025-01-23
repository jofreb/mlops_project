from pathlib import Path
import tensorflow as tf
import datetime as dt
import polars as pl
import gc
import os
import numpy as np


from src.nrms_ml_ops.utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_USER_COL,
)

from src.nrms_ml_ops.utils._articles import create_article_id_to_value_mapping

from src.nrms_ml_ops.dataloader import NRMSDataLoader
from src.nrms_ml_ops.model_config import hparams_nrms
from src.nrms_ml_ops.model import NRMSModel_docvec



os.environ["TOKENIZERS_PARALLELISM"] = "false"
tf.config.optimizer.set_jit(False)

def test_model_creation():
        model = NRMSModel_docvec(
        hparams=hparams_nrms,
        seed=42,
        )
        assert model is not None, "Model creation failed"

def test_data_loading():
    PATH = Path("./data/processed").expanduser()
    
    # chech if PATH exists
    if not PATH.exists():
        raise ValueError("Path does not exist")
    
    DUMP_DIR = PATH.joinpath("ebnerd_predictions")
    DUMP_DIR.mkdir(exist_ok=True, parents=True)
    DT_NOW = dt.datetime.now()

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    EPOCHS = 2
    embedding = "xlm_roberta_base"
    BATCH_SIZE = 32
    learning_rate = 1e-4
    HISTORY_SIZE = 35
    
    if BATCH_SIZE < 1:
        raise ValueError("Batch size is smaller than 1")

    
    
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
    
    df_train = pl.read_parquet(PATH.joinpath("train.parquet"))
    
    #check df_train dimensions
    if df_train.shape[0] == 0:
        raise ValueError("Train dataframe is empty")

    df_validation = pl.read_parquet(PATH.joinpath("validation.parquet"))
    #check df_validation dimensions
    if df_validation.shape[0] == 0:
        raise ValueError("Validation dataframe is empty")

    if len(df_train.shape) != len(df_validation.shape):
        raise ValueError("Train and validation dataframes have different shapes")

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
    
    #check if article_mapping is not empty
    if len(article_mapping) == 0:
        raise ValueError("Article mapping is empty")

    
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
    #check if train_dataloader and val_dataloader are not empty
    if len(train_dataloader) == 0:
        raise ValueError("Train dataloader is empty")
    if len(val_dataloader) == 0:
        raise ValueError("Validation dataloader is empty")
    

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
        filepath=MODEL_WEIGHTS + "nrms.weights.h5",
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
    
def test_model_training():
    PATH = Path("./data/processed").expanduser()
    
    DUMP_DIR = PATH.joinpath("ebnerd_predictions")
    DUMP_DIR.mkdir(exist_ok=True, parents=True)
    DT_NOW = dt.datetime.now()

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    EPOCHS = 2
    embedding = "xlm_roberta_base"
    BATCH_SIZE = 32
    learning_rate = 1e-4
    HISTORY_SIZE = 35
    
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
    
    df_train = pl.read_parquet(PATH.joinpath("train.parquet"))
    df_validation = pl.read_parquet(PATH.joinpath("validation.parquet"))
    
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
        filepath=MODEL_WEIGHTS + "nrms.weights.h5",
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

    tr_loss = hist.history["loss"]
    assert len(tr_loss) > 0, "Training did not run for multiple epochs"
    assert tr_loss[-1] < tr_loss[0], "Training loss did not decrease"
    
