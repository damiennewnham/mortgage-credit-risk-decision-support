from src.data import load_dataset, split_inputs
from src.config import DATA_PATH, FEATURES

def test_dataset_shape_and_target():
    df=load_dataset(DATA_PATH); X,y=split_inputs(df)
    assert len(df)==800
    assert list(X.columns)==FEATURES
    assert set(y.unique())=={0,1}
