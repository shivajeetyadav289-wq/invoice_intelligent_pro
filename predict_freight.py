from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = (
	Path(__file__).resolve().parent
	/ "fregit_cost_prediction"
	/ "models"
	/ "best_model.pkl"
)
FEATURE_COLUMNS = ["Quantity", "Dollars"]


def load_freight_model(model_path: str | Path = MODEL_PATH):
	"""Load the saved freight prediction model."""
	model_path = Path(model_path)
	if not model_path.is_file():
		raise FileNotFoundError(f"Freight model not found: {model_path}")
	return joblib.load(model_path)


def predict_freight(
	input_data: pd.DataFrame,
	model_path: str | Path = MODEL_PATH,
) -> pd.DataFrame:
	"""Predict freight cost for unseen Quantity and Dollars values."""
	missing_columns = [
		column for column in FEATURE_COLUMNS if column not in input_data.columns
	]
	if missing_columns:
		raise ValueError(f"Missing required columns: {missing_columns}")

	model = load_freight_model(model_path)
	prediction_data = input_data[FEATURE_COLUMNS].copy()
	prediction_data["Predicted_Freight"] = model.predict(prediction_data)
	return prediction_data


def main() -> None:
	"""Run an example prediction using unseen invoice values."""
	unseen_data = pd.DataFrame(
		{
			"Quantity": [125, 300],
			"Dollars": [22000, 47500],
		}
	)

	results = predict_freight(unseen_data)
	print(results.to_string(index=False))


if __name__ == "__main__":
	main()
