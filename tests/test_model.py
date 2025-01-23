from pathlib import Path
import polars as pl
import numpy as np
from tests import _PATH_DATA, _PATH_SRC
from src.nrms_ml_ops.model import NRMSModel_docvec
from src.nrms_ml_ops.model_config import hparams_nrms
from src.nrms_ml_ops.dataloader import NRMSDataLoader
from src.nrms_ml_ops.utils._articles import create_article_id_to_value_mapping 
from src.nrms_ml_ops.utils._constants import DEFAULT_HISTORY_ARTICLE_ID_COL


def test_model_output_dimensions():
    """Test that the model output has the correct dimensions."""
    # Load the model
    embedding = "xlm_roberta_base"
    BATCH_SIZE = 32
    df_test = pl.read_parquet(_PATH_DATA.joinpath("test.parquet"))
    df_articles = pl.read_parquet(_PATH_DATA.joinpath("articles.parquet"))

    precomputed_embeddings = pl.read_parquet(_PATH_DATA.joinpath(embedding + ".parquet"))

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
    
    model = NRMSModel_docvec(
        hparams= hparams_nrms,
        seed=42,
    )

    # Pass the input through the model
    output = model.scorer.predict(test_dataloader)