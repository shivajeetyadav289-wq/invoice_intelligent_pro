from pathlib import Path
import sqlite3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


DEFAULT_DATABASE_PATH = (
	Path(__file__).resolve().parents[1] / "data" / "inventory.db"
)

FEATURE_COLUMNS = [
	"days_to_pay",
	"days_po_to_invoice",
	"avg_receiving_delay",
	"quantity_difference_pct",
	"dollars_difference_pct",
	"missing_purchase_match",
	"late_invoice_flag",
	"late_payment_flag",
]

TARGET_COLUMN = "invoice_risk_label"


def load_invoice_data(
	db_path: str | Path = DEFAULT_DATABASE_PATH,
) -> pd.DataFrame:
	"""Load vendor invoices and purchase totals joined by purchase order."""
	database_path = Path(db_path)
	if not database_path.is_file():
		raise FileNotFoundError(f"Database not found: {database_path}")

	query = """
	SELECT
		vi.PONumber,
		vi.Quantity AS invoice_quantity,
		vi.Dollars AS invoice_dollars,
		vi.Freight,
		julianday(vi.InvoiceDate) - julianday(vi.PODate)
			AS days_po_to_invoice,
		julianday(vi.PayDate) - julianday(vi.InvoiceDate)
			AS days_to_pay,
		p.total_brands,
		p.total_item_quantity,
		p.total_item_dollars,
		p.avg_receiving_delay
	FROM vendor_invoice AS vi
	LEFT JOIN (
		SELECT
			PONumber,
			COUNT(DISTINCT Brand) AS total_brands,
			SUM(Quantity) AS total_item_quantity,
			SUM(Dollars) AS total_item_dollars,
			AVG(
				julianday(ReceivingDate) - julianday(PODate)
			) AS avg_receiving_delay
		FROM purchases
		GROUP BY PONumber
	) AS p
		ON vi.PONumber = p.PONumber
	"""

	with sqlite3.connect(database_path) as connection:
		return pd.read_sql_query(query, connection)


def invoice_risk_label(
	row: pd.Series,
	quantity_tolerance: float = 0.10,
	dollars_tolerance: float = 0.10,
	late_invoice_days: int = 30,
	late_payment_days: int = 30,
) -> str:
	"""Assign High, Medium, or Low risk to one merged invoice row."""
	risk_reasons = []

	if pd.isna(row["total_item_quantity"]):
		risk_reasons.append("missing_purchase_match")
	elif row["total_item_quantity"] == 0:
		risk_reasons.append("zero_purchase_quantity")
	elif (
		abs(row["invoice_quantity"] - row["total_item_quantity"])
		/ row["total_item_quantity"]
		> quantity_tolerance
	):
		risk_reasons.append("quantity_mismatch")

	if pd.isna(row["total_item_dollars"]):
		risk_reasons.append("missing_purchase_match")
	elif row["total_item_dollars"] == 0:
		risk_reasons.append("zero_purchase_dollars")
	elif (
		abs(row["invoice_dollars"] - row["total_item_dollars"])
		/ row["total_item_dollars"]
		> dollars_tolerance
	):
		risk_reasons.append("dollars_mismatch")

	if (
		pd.notna(row["days_po_to_invoice"])
		and row["days_po_to_invoice"] > late_invoice_days
	):
		risk_reasons.append("late_invoice")

	if (
		pd.notna(row["days_to_pay"])
		and row["days_to_pay"] > late_payment_days
	):
		risk_reasons.append("late_payment")

	if len(risk_reasons) >= 2:
		return "High"
	if risk_reasons:
		return "Medium"
	return "Low"


def prepare_features(
	data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
	"""Create engineered features X and the invoice risk target y."""
	model_data = data.copy()

	required_columns = {
		"invoice_quantity",
		"invoice_dollars",
		"total_item_quantity",
		"total_item_dollars",
		"days_po_to_invoice",
		"days_to_pay",
		"avg_receiving_delay",
	}
	missing_columns = sorted(required_columns - set(model_data.columns))
	if missing_columns:
		raise ValueError(f"Missing required columns: {missing_columns}")

	purchase_quantity = model_data["total_item_quantity"].replace(0, pd.NA)
	purchase_dollars = model_data["total_item_dollars"].replace(0, pd.NA)

	model_data["quantity_difference_pct"] = (
		(
			model_data["invoice_quantity"]
			- model_data["total_item_quantity"]
		).abs()
		/ purchase_quantity
	).fillna(1.0)

	model_data["dollars_difference_pct"] = (
		(
			model_data["invoice_dollars"]
			- model_data["total_item_dollars"]
		).abs()
		/ purchase_dollars
	).fillna(1.0)

	model_data["missing_purchase_match"] = (
		model_data["total_item_quantity"].isna()
		| model_data["total_item_dollars"].isna()
	).astype(int)
	model_data["late_invoice_flag"] = (
		model_data["days_po_to_invoice"] > 30
	).fillna(False).astype(int)
	model_data["late_payment_flag"] = (
		model_data["days_to_pay"] > 30
	).fillna(False).astype(int)

	model_data[TARGET_COLUMN] = model_data.apply(
		invoice_risk_label,
		axis=1,
	)

	X = model_data[FEATURE_COLUMNS].copy()
	y = model_data[TARGET_COLUMN].copy()
	return X, y


def split_and_scale_data(
	X: pd.DataFrame,
	y: pd.Series,
	test_size: float = 0.20,
	random_state: int = 42,
) -> tuple[
	pd.DataFrame,
	pd.DataFrame,
	pd.Series,
	pd.Series,
	MinMaxScaler,
]:
	"""Stratify, split, and MinMax-scale the model features."""
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=test_size,
		random_state=random_state,
		stratify=y,
	)

	scaler = MinMaxScaler()
	X_train_scaled = pd.DataFrame(
		scaler.fit_transform(X_train),
		columns=X_train.columns,
		index=X_train.index,
	)
	X_test_scaled = pd.DataFrame(
		scaler.transform(X_test),
		columns=X_test.columns,
		index=X_test.index,
	)

	return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def load_prepare_and_split(
	db_path: str | Path = DEFAULT_DATABASE_PATH,
) -> tuple[
	pd.DataFrame,
	pd.DataFrame,
	pd.Series,
	pd.Series,
	MinMaxScaler,
]:
	"""Run the complete notebook preprocessing workflow."""
	data = load_invoice_data(db_path)
	X, y = prepare_features(data)
	return split_and_scale_data(X, y)
