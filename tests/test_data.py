from torch.utils.data import Dataset
import polars as pl
from tests import _PATH_DATA
from pathlib import Path

def test_my_dataset():
    """Test the MyDataset class."""
    path_data = Path(_PATH_DATA)
    train = pl.read_parquet(path_data.joinpath("train.parquet"))
    test = pl.read_parquet(path_data.joinpath("test.parquet"))
    validation = pl.read_parquet(path_data.joinpath("validation.parquet"))
    assert len(train) == 3513
    assert len(test) == 2466
    assert len(validation) == 765
    # for dataset in [train, test]:
    #     for x, y in dataset:
    #         assert x.shape == (1, 28, 28)
    #         assert y in range(10)
    # train_targets = torch.unique(train.tensors[1])
    # assert (train_targets == torch.arange(0,10)).all()
    # test_targets = torch.unique(test.tensors[1])
    # assert (test_targets == torch.arange(0,10)).all()