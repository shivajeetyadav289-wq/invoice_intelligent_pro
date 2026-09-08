import json
import sys
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

if __package__:
	from .data_preprocessing import (
		DEFAULT_DATABASE_PATH,
		load_vendor_invoice_data,
		prepare_feature,
		split_data,
	)
else:
	sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
	from fregit_cost_prediction.data_preprocessing import (
		DEFAULT_DATABASE_PATH,
		load_vendor_invoice_data,
		prepare_feature,
		split_data,
	)


DEFAULT_MODEL_DIRECTORY = Path(__file__).parent / "models"


def train_models(X_train, y_train) -> dict[str, Any]:
	"""Train the three regression models used in the notebook."""
	models = {
		"linear_regression": LinearRegression(),
		"decision_tree": DecisionTreeRegressor(max_depth=4, random_state=42),
		"random_forest": RandomForestRegressor(max_depth=5, random_state=42),
	}

	for model in models.values():
		model.fit(X_train, y_train)

	return models


def evaluate_models(models, X_test, y_test) -> dict[str, float]:
	"""Calculate test MAE for every fitted model."""
	return {
		name: mean_absolute_error(y_test, model.predict(X_test))
		for name, model in models.items()
	}


def save_best_model(
	model: Any,
	model_name: str,
	mae: float,
	model_directory: str | Path = DEFAULT_MODEL_DIRECTORY,
) -> Path:
	"""Save the best model and its selection metadata."""
	directory = Path(model_directory)
	directory.mkdir(parents=True, exist_ok=True)

	model_path = directory / "best_model.pkl"
	joblib.dump(model, model_path)

	metadata = {
		"model_name": model_name,
		"mae": mae,
	}
	(directory / "best_model_metadata.json").write_text(
		json.dumps(metadata, indent=2),
		encoding="utf-8",
	)
	return model_path


def load_model(
	model_directory: str | Path = DEFAULT_MODEL_DIRECTORY,
) -> Any:
	"""Load the saved best model."""
	model_path = Path(model_directory) / "best_model.pkl"
	if not model_path.is_file():
		raise FileNotFoundError(f"Model not found: {model_path}")

	return joblib.load(model_path)


def main() -> dict[str, Any]:
	"""Run the complete training pipeline and save the lowest-MAE model."""
	data = load_vendor_invoice_data(DEFAULT_DATABASE_PATH)
	X, y = prepare_feature(data)
	X_train, X_test, y_train, y_test = split_data(X, y)

	models = train_models(X_train, y_train)
	metrics = evaluate_models(models, X_test, y_test)
	best_name = min(metrics, key=metrics.get)
	best_path = save_best_model(
		models[best_name],
		best_name,
		metrics[best_name],
	)

	print("Test MAE:", metrics)
	print(f"Best model: {best_name} (MAE: {metrics[best_name]:.4f})")
	print(f"Saved to: {best_path}")

	return {
		"models": models,
		"metrics": metrics,
		"best_model_name": best_name,
		"best_model_path": best_path,
	}


if __name__ == "__main__":
	main()
