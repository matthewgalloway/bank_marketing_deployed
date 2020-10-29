from prediction_model.config import config

import pandas as pd


def validate_inputs(input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for unprocessable values."""

    validated_data = input_data.copy()

    if input_data[config.FEATURES].isnull().any().any():
        validated_data = validated_data.dropna(
            axis=0)
    return validated_data
