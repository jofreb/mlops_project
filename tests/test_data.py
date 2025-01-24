
#from torch.utils.data import Dataset
import polars as pl
from tests import _PATH_DATA
from pathlib import Path

def test_dataset_dimensions():
    """Test that the dataset has the correct dimensions."""
    path_data = Path(_PATH_DATA)
    train = pl.read_parquet(path_data.joinpath("train.parquet"))
    test = pl.read_parquet(path_data.joinpath("test.parquet"))
    validation = pl.read_parquet(path_data.joinpath("validation.parquet"))
    assert len(train) == 3513
    assert len(test) == 2466
    assert len(validation) == 765
    for dataset in [train, test, validation]:
        for sample in range(len(dataset)):
            assert dataset[sample].shape == (1,6)
            

# write a test that checks that there are no missing values in the dataset

def test_dataset_missing_values():
    """Test that the dataset has no missing values."""
    path_data = Path(_PATH_DATA)
    train = pl.read_parquet(path_data.joinpath("train.parquet"))
    test = pl.read_parquet(path_data.joinpath("test.parquet"))
    validation = pl.read_parquet(path_data.joinpath("validation.parquet"))

    assert train.null_count().sum_horizontal()[0] == 0
    assert test.null_count().sum_horizontal()[0] == 0
    assert validation.null_count().sum_horizontal()[0] == 0  
    
# write a test that checks that the dataset has the correct columns
def test_dataset_columns():
    """Test that the dataset has the correct columns."""
    path_data = Path(_PATH_DATA)
    train = pl.read_parquet(path_data.joinpath("train.parquet"))
    test = pl.read_parquet(path_data.joinpath("test.parquet"))
    validation = pl.read_parquet(path_data.joinpath("validation.parquet"))
    assert train.columns == ['user_id', 'article_id_fixed', 'article_ids_inview', 'article_ids_clicked', 'impression_id', 'labels']
    assert test.columns == ['user_id', 'article_id_fixed', 'article_ids_inview', 'article_ids_clicked', 'impression_id', 'labels']
    assert validation.columns == ['user_id', 'article_id_fixed', 'article_ids_inview', 'article_ids_clicked', 'impression_id', 'labels']  

