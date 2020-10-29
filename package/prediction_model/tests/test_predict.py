import math

from prediction_model.predict import make_prediction
from prediction_model.config import config
import pandas as pd


def test_make_single_prediction():
    # Given
    test_data = pd.read_csv(config.TEST_DATASET_DIR, ",")
    single_test_input = test_data

    # When
    subject = make_prediction(input_data=single_test_input[config.FEATURES])

    # Then
    assert subject is not None
    assert isinstance(subject.get('predictions')[0], float)
    assert math.ceil(subject.get('predictions')[0]) == 1


# def test_make_multiple_predictions():
#     # Given
#     test_data = pd.read_csv(config.TEST_DATASET_DIR, ",")
#     original_data_length = len(test_data)
#     multiple_test_input = test_data
#
#     # When
#     subject = make_prediction(input_data=multiple_test_input)
#
#     # Then
#     assert subject is not None
#     assert len(subject.get('predictions')) == 9043
#
#     # We expect some rows to be filtered out
#     assert len(subject.get('predictions')) != original_data_length