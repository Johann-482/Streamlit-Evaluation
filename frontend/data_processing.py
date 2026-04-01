import config
from backend.preprocessing import preprocess_all


def prepare_data(df, missing_rate):

    config.TRAIN_MISSING_RATE = missing_rate
    config.TEST_MISSING_RATE = missing_rate

    return preprocess_all(df)
