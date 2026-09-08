from typing import Any

import pandas as pd
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
)


def _evaluate_classifier(
	model: Any,
	X_test: pd.DataFrame,
	y_test: pd.Series,
	model_name: str,
) -> dict[str, Any]:
	"""Evaluate one fitted classifier."""
	predictions = model.predict(X_test)
	labels = sorted(y_test.unique())

	return {
		"Model": model_name,
		"Accuracy": accuracy_score(y_test, predictions),
		"Precision": precision_score(
			y_test,
			predictions,
			average="macro",
			zero_division=0,
		),
		"Recall": recall_score(
			y_test,
			predictions,
			average="macro",
			zero_division=0,
		),
		"F1": f1_score(
			y_test,
			predictions,
			average="macro",
			zero_division=0,
		),
		"Classification_Report": classification_report(
			y_test,
			predictions,
			zero_division=0,
		),
		"Confusion_Matrix": confusion_matrix(
			y_test,
			predictions,
			labels=labels,
		),
		"Labels": labels,
	}


def evaluate_logistic_regression(
	model: Any,
	X_test: pd.DataFrame,
	y_test: pd.Series,
) -> dict[str, Any]:
	"""Evaluate the fitted logistic regression classifier."""
	return _evaluate_classifier(
		model,
		X_test,
		y_test,
		"Logistic Regression",
	)


def evaluate_decision_tree(
	model: Any,
	X_test: pd.DataFrame,
	y_test: pd.Series,
) -> dict[str, Any]:
	"""Evaluate the fitted decision tree classifier."""
	return _evaluate_classifier(
		model,
		X_test,
		y_test,
		"Decision Tree Classifier",
	)


def evaluate_random_forest(
	model: Any,
	X_test: pd.DataFrame,
	y_test: pd.Series,
) -> dict[str, Any]:
	"""Evaluate the fitted random forest classifier."""
	return _evaluate_classifier(
		model,
		X_test,
		y_test,
		"Random Forest Classifier",
	)


def evaluate_models(
	models: dict[str, Any],
	X_test: pd.DataFrame,
	y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
	"""Evaluate all fitted classifiers and return summary and details."""
	evaluators = {
		"logistic_regression": evaluate_logistic_regression,
		"decision_tree": evaluate_decision_tree,
		"random_forest": evaluate_random_forest,
	}

	details = {}
	summary = []

	for model_key, model in models.items():
		evaluator = evaluators.get(model_key, _evaluate_classifier)
		if evaluator is _evaluate_classifier:
			result = evaluator(model, X_test, y_test, model_key)
		else:
			result = evaluator(model, X_test, y_test)

		details[model_key] = result
		summary.append(
			{
				"Model": result["Model"],
				"Accuracy": result["Accuracy"],
				"Precision": result["Precision"],
				"Recall": result["Recall"],
				"F1": result["F1"],
			}
		)

	summary_df = pd.DataFrame(summary).sort_values(
		"F1",
		ascending=False,
	).reset_index(drop=True)
	return summary_df, details
