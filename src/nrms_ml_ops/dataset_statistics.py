#import numpy as np
import polars as pl
import typer


from dataloader import NRMSDataLoader
from utils._constants import DEFAULT_HISTORY_ARTICLE_ID_COL, DEFAULT_LABELS_COL, DEFAULT_INVIEW_ARTICLES_COL
from utils._articles import create_article_id_to_value_mapping


def dataset_statistics(dataset_path: str):
    """
    Compute dataset statistics.
    """

    df_train = pl.read_parquet(f"{dataset_path}train.parquet")
    df_validation= pl.read_parquet(f"{dataset_path}validation.parquet")
    df_articles = pl.read_parquet(f"{dataset_path}articles.parquet")

    # Load the dataset using Polars
    print(f"Loading dataset from: {dataset_path}")

    embedding = 'xlm_roberta_base'
    precomputed_embeddings = pl.read_parquet(f"{dataset_path}{embedding}.parquet")

    precomputed_embeddings = precomputed_embeddings.filter(precomputed_embeddings['article_id'].is_in(df_articles['article_id']))
    precomputed_embeddings = precomputed_embeddings.rename({'FacebookAI/xlm-roberta-base': 'embedding'})

    #pre_embs = np.array([precomputed_embeddings['embedding'][0]])

    article_mapping = create_article_id_to_value_mapping(
        df=precomputed_embeddings,
        value_col="embedding",  # Column containing precomputed embeddings
        article_col="article_id",  # Column containing article IDs
    )

    # Initialize the NRMSDataLoader
    train_dataloader = NRMSDataLoader(
        behaviors=df_train,
        article_dict=article_mapping,
        unknown_representation="zeros",
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        eval_mode=False,
        #batch_size=BATCH_SIZE,
    )

    test_dataloader = NRMSDataLoader(
        behaviors=df_validation,
        article_dict=article_mapping,
        unknown_representation="zeros",
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        eval_mode=True,
        #batch_size=BATCH_SIZE,
    )

    # Total number of samples
    total_samples_train = len(train_dataloader.X)
    print(f"Total samples train dataset: {total_samples_train}")
    total_samples_test = len(test_dataloader.X)
    print(f"Total samples test dataset: {total_samples_test}")

    # Label distribution
    label_counts_train = (
        train_dataloader.y.to_frame()
        .groupby(DEFAULT_LABELS_COL)
        .count()
        .to_dict(as_series=False)
    )
    label_counts_test = (
        train_dataloader.y.to_frame()
        .groupby(DEFAULT_LABELS_COL)
        .count()
        .to_dict(as_series=False)
    )

    # Average articles in history
    avg_history_len_train = train_dataloader.X[DEFAULT_HISTORY_ARTICLE_ID_COL].list.len().mean()
    avg_history_len_test = test_dataloader.X[DEFAULT_HISTORY_ARTICLE_ID_COL].list.len().mean()

    print(f"Average number of train articles in user history: {avg_history_len_train:.2f}")
    print(f"Average number of test articles in user history: {avg_history_len_test:.2f}")

    # Average articles in view
    avg_inview_len = train_dataloader.X[DEFAULT_INVIEW_ARTICLES_COL].list.len().mean()
    print(f"Average number of articles in view: {avg_inview_len:.2f}")

    print("Sample dataset:",train_dataloader.X)
    print("Dataset columns:", train_dataloader.X.columns)

    print("\nStatistics computation complete.")



if __name__ == "__main__":
    typer.run(dataset_statistics)