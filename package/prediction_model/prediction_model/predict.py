import numpy as np
import pandas as pd

from prediction_model.processing.data_management import load_pipeline
from prediction_model.config import config
from prediction_model import __version__ as _version

import logging
import typing as t


_logger = logging.getLogger(__name__)

pipeline_file_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"
_prediction_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data: t.Union[pd.DataFrame, dict],
                    ) -> dict:
    """Make a prediction using a saved model pipeline.

    Args:
        input_data: Array of model prediction inputs.

    Returns:
        Predictions for each input row, as well as the model version.
    """

    data = pd.DataFrame(input_data)

    prediction = _prediction_pipe.predict(data[config.FEATURES])

    output = np.exp(prediction)

    results = {"predictions": output, "version": _version}

    _logger.info(
        f"Making predictions with model version: {_version} "
        f"Inputs: {data} "
        f"Predictions: {results}"
    )

    return results
