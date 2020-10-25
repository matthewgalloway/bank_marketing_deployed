
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class NumericalMinMaxNormalisation(BaseEstimator, TransformerMixin):
	"""Applies min max scaling to numerical values """
	def __init__(self, variables=None) -> None:
		if not isinstance(variables, list):
			self.variables = [variables]
		else:
			self.variables = variables
	def fit(self, X: pd.DataFrame, y: pd.Series=None)-> "NumericalMinMaxNormalisation":
		""""fit statement to intergrate with Sklearn pipeline"""
		return self

	def transform(self, X: pd.DataFrame) -> pd.DataFrame:
		"""Apply the transforms to the dataframe."""

		for variable in self.varibles:
			X[variable] = (X[variable] - min(X[variable])) / max(X[variable])


class CategoricalImputer(BaseEstimator, TransformerMixin):
    """Categorical data missing value imputer."""

    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "CategoricalImputer":
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna("Missing")

        return X
