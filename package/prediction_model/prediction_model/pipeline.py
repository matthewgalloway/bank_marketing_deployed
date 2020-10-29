from sklearn.pipeline import Pipeline
from prediction_model.config import config
from prediction_model.preprocessing import preprocessors as pp
from sklearn.ensemble import RandomForestClassifier


import logging
_logger = logging.getLogger(__name__)


prediction_pipe = Pipeline(
    [
        (
            "Min_Max_Numerical",
            pp.NumericalMinMaxNormalisation(variables=config.NUMERICALS__VARS),
        ),
        (
            "Categorical_Dummies",
            pp.CategoricalDummyEncoder(variables=config.CATEGORICAL_VARS),
        ),
        (
            "RF_model", RandomForestClassifier(min_samples_leaf=1, random_state=0)
        )
    ]
)

