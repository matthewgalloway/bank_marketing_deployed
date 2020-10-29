import pathlib

import prediction_model

PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "datasets"

# numerical variables
NUMERICALS__VARS = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

# categorical variables
CATEGORICAL_VARS = [
	"job",
	"marital",
	"education",
	"default",
	"housing",
	"loan",
	"contact",
	"month",
	"poutcome",
]

TRAIN_DATASET_DIR = DATASET_DIR/'train.csv'

FEATURES = ['age',
			'job',
			'marital',
			'education',
			'default',
			'balance',
			'housing',
			'loan',
			'contact',
			'day',
			'month',
			'duration',
			'campaign',
			'pdays',
			'previous',
			'poutcome']

TARGET = ['y']

PIPELINE_NAME = "RF_Pipeline"
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output_v"

