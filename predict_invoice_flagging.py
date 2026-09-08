from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = (
	Path(__file__).resolve().parent
	/ "Invoice_flagging"
	/ "models"
	/ "best_model.joblib"
)


def load_invoice_flagging_model(model_path: str | Path = MODEL_PATH) -> dict:
	"""Load the saved invoice-risk model bundle."""
	model_path = Path(model_path)
	if not model_path.is_file():
		raise FileNotFoundError(f"Invoice flagging model not found: {model_path}")
	return joblib.load(model_path)


def build_prediction_features(unseen_data: pd.DataFrame) -> pd.DataFrame:
	"""Create the features expected by the saved invoice-risk model."""
	required_columns = [
		"invoice_quantity",
		"invoice_dollars",
		"total_item_quantity",
		"total_item_dollars",
		"days_po_to_invoice",
		"days_to_pay",
		"avg_receiving_delay",
	]
	missing_columns = [
		column for column in required_columns if column not in unseen_data.columns
	]
	if missing_columns:
		raise ValueError(f"Missing required columns: {missing_columns}")

	data = unseen_data.copy()
	purchase_quantity = data["total_item_quantity"].replace(0, pd.NA)
	purchase_dollars = data["total_item_dollars"].replace(0, pd.NA)

	data["quantity_difference_pct"] = (
		(
			data["invoice_quantity"]
			- data["total_item_quantity"]
		).abs()
		/ purchase_quantity
	).fillna(1.0)
	data["dollars_difference_pct"] = (
		(
			data["invoice_dollars"]
			- data["total_item_dollars"]
		).abs()
		/ purchase_dollars
	).fillna(1.0)
	data["missing_purchase_match"] = (
		data["total_item_quantity"].isna()
		| data["total_item_dollars"].isna()
	).astype(int)
	data["late_invoice_flag"] = (
		data["days_po_to_invoice"] > 30
	).fillna(False).astype(int)
	data["late_payment_flag"] = (
		data["days_to_pay"] > 30
	).fillna(False).astype(int)

	feature_names = [
		"days_to_pay",
		"days_po_to_invoice",
		"avg_receiving_delay",
		"quantity_difference_pct",
		"dollars_difference_pct",
		"missing_purchase_match",
		"late_invoice_flag",
		"late_payment_flag",
	]
	return data[feature_names]


def predict_invoice_risk(
	unseen_data: pd.DataFrame,
	model_path: str | Path = MODEL_PATH,
) -> pd.DataFrame:
	"""Predict risk labels for unseen invoice records."""
	bundle = load_invoice_flagging_model(model_path)
	features = build_prediction_features(unseen_data)
	feature_names = bundle["feature_names"]
	features = features[feature_names]
	scaled_features = pd.DataFrame(
		bundle["scaler"].transform(features),
		columns=feature_names,
		index=features.index,
	)

	results = unseen_data.copy()
	results["Predicted_Risk"] = bundle["model"].predict(scaled_features)
	return results


def main() -> None:
	"""Run an example prediction using unseen invoice data."""
	unseen_data = pd.DataFrame(
		{
			"invoice_quantity": [1000, 5000],
			"invoice_dollars": [12000, 80000],
			"total_item_quantity": [1000, 3000],
			"total_item_dollars": [12000, 50000],
			"days_po_to_invoice": [10, 45],
			"days_to_pay": [20, 60],
			"avg_receiving_delay": [8.0, 12.0],
		}
	)

	results = predict_invoice_risk(unseen_data)
	print(results.to_string(index=False))


if __name__ == "__main__":
	main()
