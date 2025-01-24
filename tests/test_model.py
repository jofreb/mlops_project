# write a test that checks that from a input dataset, the output dataset has the correct dimensions

# def test_model_output_dimensions():
#     """Test that the model output has the correct dimensions."""
#     path_data = Path(_PATH_DATA)
#     train = pl.read_parquet(path_data.joinpath("train.parquet"))
#     x = train[0]
#     y = model(x)
#     assert y.shape == (1, 1)