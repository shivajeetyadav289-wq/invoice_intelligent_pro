import json
import sys
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


if __package__:
	from .data_preprocessing import (
		DEFAULT_DATABASE_PATH,
		load_prepare_and_split,
	)
	from .model_evaluation import evaluate_models
else:
	sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
	from Invoice_flagging.data_preprocessing import (
		DEFAULT_DATABASE_PATH,
		load_prepare_and_split,
	)
	from Invoice_flagging.model_evaluation import evaluate_models


DEFAULT_MODEL_DIRECTORY = Path(__file__).resolve().parent / "models"


def train_models(X_train, y_train) -> dict[str, Any]:
	"""Train the three classifiers used in the invoice-risk notebook."""
	models = {
		"logistic_regression": LogisticRegression(
			max_iter=1000,
			random_state=42,
		),
		"decision_tree": DecisionTreeClassifier(
			max_depth=5,
			random_state=42,
		),
		"random_forest": RandomForestClassifier(
			n_estimators=100,
			max_depth=5,
			random_state=42,
		),
	}

	for model in models.values():
		model.fit(X_train, y_train)

	return models


def save_best_model(
	model: Any,
	model_name: str,
	metrics: dict[str, Any],
	scaler: Any,
	feature_names: list[str],
	model_directory: str | Path = DEFAULT_MODEL_DIRECTORY,
) -> Path:
	"""Save the best model with its scaler and prediction metadata."""
	directory = Path(model_directory)
	directory.mkdir(parents=True, exist_ok=True)

	model_path = directory / "best_model.joblib"
	joblib.dump(
		{
			"model": model,
			"scaler": scaler,
			"feature_names": feature_names,
			"model_name": model_name,
		},
		model_path,
	)

	metadata = {
		"model_name": model_name,
		"accuracy": float(metrics["Accuracy"]),
		"precision": float(metrics["Precision"]),
		"recall": float(metrics["Recall"]),
		"f1": float(metrics["F1"]),
	}
	(directory / "best_model_metadata.json").write_text(
		json.dumps(metadata, indent=2),
		encoding="utf-8",
	)
	return model_path


def load_best_model(
	model_directory: str | Path = DEFAULT_MODEL_DIRECTORY,
) -> dict[str, Any]:
	"""Load the saved model bundle."""
	model_path = Path(model_directory) / "best_model.joblib"
	if not model_path.is_file():
		raise FileNotFoundError(f"Model not found: {model_path}")
	return joblib.load(model_path)


def main() -> dict[str, Any]:
	"""Run preprocessing, training, evaluation, and best-model saving."""
	X_train, X_test, y_train, y_test, scaler = load_prepare_and_split(
		DEFAULT_DATABASE_PATH,
	)
	models = train_models(X_train, y_train)
	results_df, details = evaluate_models(models, X_test, y_test)

	best_model_name = results_df.iloc[0]["Model"]
	model_key_by_label = {
		"Logistic Regression": "logistic_regression",
		"Decision Tree Classifier": "decision_tree",
		"Random Forest Classifier": "random_forest",
	}
	best_model_key = model_key_by_label[best_model_name]
	best_metrics = details[best_model_key]
	best_model_path = save_best_model(
		models[best_model_key],
		best_model_key,
		best_metrics,
		scaler,
		list(X_train.columns),
	)

	print(results_df.to_string(index=False))
	print(f"Best model: {best_model_key}")
	print(f"Saved to: {best_model_path}")

	return {
		"models": models,
		"results": results_df,
		"details": details,
		"best_model_name": best_model_key,
		"best_model_path": best_model_path,
	}


if __name__ == "__main__":
	main()
