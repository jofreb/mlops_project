import json
from contextlib import asynccontextmanager
from pathlib import Path
import os
import gc
import datetime as dt

import anyio
import tensorflow as tf
import polars as pl
from fastapi import FastAPI, HTTPException, File, UploadFile

from utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_USER_COL,
)
from utils._behaviors import add_prediction_scores
from utils._articles import create_article_id_to_value_mapping
from evaluation import AucScore
from dataloader import NRMSDataLoader
from model_config import hparams_nrms
from model import NRMSModel_docvec
from google.cloud import storage

# FastAPI application
app = FastAPI()

# Model and configuration variables
MODEL_WEIGHTS_LOCAL = Path("/tmp/nrms.weights.h5")
CLOUD_PATH = "gs://project_mlops_bucket/data/processed/"
EMBEDDING = "xlm_roberta_base"
BATCH_SIZE = 32
HISTORY_SIZE = 35
MODEL = None
ARTICLE_MAPPING = None
TEST_DATALOADER = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and clean up resources."""
    global MODEL, ARTICLE_MAPPING, TEST_DATALOADER

    # Download model weights
    def download_blob(bucket_name, source_blob_name, destination_file_name):
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

    download_blob("project_mlops_bucket", "models/nrms.weights.h5", str(MODEL_WEIGHTS_LOCAL))

    if not MODEL_WEIGHTS_LOCAL.exists():
        raise FileNotFoundError(f"Model weights not found at {MODEL_WEIGHTS_LOCAL}")

    # Set up GPU memory growth
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Load datasets and embeddings
    df_test = pl.read_parquet(CLOUD_PATH + "test.parquet")
    df_articles = pl.read_parquet(CLOUD_PATH + "articles.parquet")
    precomputed_embeddings = pl.read_parquet(CLOUD_PATH + EMBEDDING + ".parquet")

    precomputed_embeddings = precomputed_embeddings.filter(
        precomputed_embeddings["article_id"].is_in(df_articles["article_id"])
    ).rename({"FacebookAI/xlm-roberta-base": "embedding"})

    ARTICLE_MAPPING = create_article_id_to_value_mapping(
        df=precomputed_embeddings,
        value_col="embedding",
        article_col="article_id",
    )

    TEST_DATALOADER = NRMSDataLoader(
        behaviors=df_test,
        article_dict=ARTICLE_MAPPING,
        unknown_representation="zeros",
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        eval_mode=True,
        batch_size=BATCH_SIZE,
    )

    hparams_nrms.history_size = HISTORY_SIZE
    hparams_nrms.title_size = precomputed_embeddings["embedding"][0].shape[0]

    # Initialize model
    MODEL = NRMSModel_docvec(hparams=hparams_nrms, seed=42)
    MODEL.model.compile(
        optimizer=MODEL.model.optimizer,
        loss=MODEL.model.loss,
        metrics=["AUC"],
    )
    MODEL.model.load_weights(str(MODEL_WEIGHTS_LOCAL))
    print("Model loaded successfully.")

    yield

    # Cleanup
    gc.collect()
    del MODEL
    del ARTICLE_MAPPING
    del TEST_DATALOADER

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to the NRMS prediction API!"}

@app.post("/predict/")
async def predict_behavior(file: UploadFile = File(...)):
    try:
        # Read the uploaded file contents
        contents = await file.read()
        if not contents:
            raise ValueError("Uploaded file is empty.")

        # Define the temporary file path
        temp_file_path = Path(f"/tmp/{file.filename}")
        
        # Write the file contents to disk
        async with await anyio.open_file(temp_file_path, "wb") as f:
            await f.write(contents)  # Use 'await' here to properly write contents
        
        # Check if the saved file is valid
        print(f"File size: {temp_file_path.stat().st_size}")
        if temp_file_path.stat().st_size < 12:
            raise ValueError("Uploaded file is too small to be a valid Parquet file.")

        # Validate Parquet file format
        df_test = pl.read_parquet(temp_file_path)
        print(f"Loaded DataFrame: {df_test}")

        # Predictions
        pred_test = MODEL.scorer.predict(TEST_DATALOADER)
        df_test = add_prediction_scores(df_test, pred_test.tolist())

        # AUC calculation
        aucsc = AucScore()
        auc = aucsc.calculate(
            y_true=df_test["labels"].to_list(),
            y_pred=df_test["scores"].to_list(),
        )

        # return {
        #     "message": "Prediction successful",
        #     "auc": auc,
        #     "predictions": df_test.select(["impression_id", "scores"]).to_dicts(),
        # }
        return {"Test AUC": auc}  # Return only the AUC value

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
