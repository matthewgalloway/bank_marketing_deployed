import pandas as pd
from sklearn.model_selection import train_test_split
from prediction_model.config import config
from prediction_model import pipeline
from prediction_model.preprocessing.data_management import save_pipeline

import logging

_logger = logging.getLogger(__name__)


def run_training():
	data = pd.read_csv(config.TRAIN_DATASET_DIR, sep=",")
	data['y'] = data['y'].eq('yes').mul(1)

	X_train, X_test, y_train, y_test = train_test_split(data[config.FEATURES], data[config.TARGET], test_size=0.1,
														random_state=0)

	pipeline.prediction_pipe.fit(X_train, y_train.values.ravel())
	# _logger.info(f'saving model version: {_version}')

	save_pipeline(pipeline_to_persist=pipeline.prediction_pipe)


if __name__ == "__main__":
	run_training()
