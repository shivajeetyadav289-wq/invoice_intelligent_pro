from pathlib import Path
import sqlite3

import pandas as pd
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = ["Quantity", "Dollars"]
TARGET_COLUMN = "Freight"
DEFAULT_DATABASE_PATH = Path(__file__).resolve().parents[1] / "data" / "inventory.db"


def load_vendor_invoice_data(
	db_path: str | Path = DEFAULT_DATABASE_PATH,
	table_name: str = "vendor_invoice",
) -> pd.DataFrame:
	"""Load the vendor invoice table from the project's SQLite database."""
	if not table_name.replace("_", "").isalnum():
		raise ValueError(f"Invalid table name: {table_name!r}")

	database_path = Path(db_path)
	if not database_path.is_file():
		raise FileNotFoundError(f"Database not found: {database_path}")

	with sqlite3.connect(database_path) as connection:
		data = pd.read_sql_query(
			f'SELECT * FROM "{table_name}"',
			connection,
		)

	return data


def prepare_feature(
	data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
	"""Return model features X and target y for freight prediction."""
	required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
	missing_columns = [
		column for column in required_columns if column not in data.columns
	]
	if missing_columns:
		raise ValueError(f"Missing required columns: {missing_columns}")

	X = data[FEATURE_COLUMNS].copy()
	y = data[TARGET_COLUMN].copy()
	return X, y


def split_data(
	X: pd.DataFrame,
	y: pd.Series,
	test_size: float = 0.2,
	random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
	"""Split X and y into X_train, X_test, y_train, and y_test."""
	return train_test_split(
		X,
		y,
		test_size=test_size,
		random_state=random_state,
	)
