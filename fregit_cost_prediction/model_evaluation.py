from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _calculate_metrics(model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> dict[str, float | str]:
	predictions = model.predict(X_test)
	mse = mean_squared_error(y_test, predictions)

	return {
		"Model": model_name,
		"MAE": mean_absolute_error(y_test, predictions),
		"RMSE": np.sqrt(mse),
		"R2": r2_score(y_test, predictions),
	}


def evaluate_linear_regression(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | str]:
	"""Evaluate the fitted linear regression model."""
	return _calculate_metrics(
		model,
		X_test,
		y_test,
		"Linear Regression",
	)


def evaluate_decision_tree(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | str]:
	"""Evaluate the fitted decision tree regressor."""
	return _calculate_metrics(
		model,
		X_test,
		y_test,
		"Decision Tree Regressor",
	)


def evaluate_random_forest(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | str]:
	"""Evaluate the fitted random forest regressor."""
	return _calculate_metrics(
		model,
		X_test,
		y_test,
		"Random Forest Regressor",
	)
