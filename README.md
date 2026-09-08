# Invoice Intelligent Pro

An invoice analytics project with two machine-learning workflows:

- Freight-cost prediction from invoice quantity and dollars.
- Vendor-invoice risk flagging for manual review.

The project also includes an interactive Streamlit dashboard.

## Features

- Loads invoice and purchase data from SQLite.
- Predicts freight cost for unseen invoices.
- Labels invoices as `Low`, `Medium`, or `High` risk.
- Trains Logistic Regression, Decision Tree, and Random Forest classifiers.
- Evaluates accuracy, precision, recall, F1 score, and confusion matrices.
- Saves the best trained models with Joblib.
- Supports single-record and batch CSV predictions in Streamlit.

## Project Structure

```text
.
├── app.py
├── predict_freight.py
├── predict_invoice_flagging.py
├── data/
│   ├── inventory.db
│   └── invoices.db
├── fregit_cost_prediction/
│   ├── data_preprocessing.py
│   ├── train.py
│   └── models/
├── Invoice_flagging/
│   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   ├── train.py
│   └── models/
└── Notebooks/
	├── Predicting_freight_cost.ipynb
	└── invoice_flagging.ipynb
```

## Requirements

Use Python 3.11 or a compatible Python 3 environment. Install the dependencies with:

```bash
python -m pip install pandas scikit-learn joblib streamlit scipy seaborn matplotlib jupyter
```

On this Windows setup, the Anaconda interpreter can be used as follows:

```bash
C:\Users\91988\anaconda3\python.exe -m pip install pandas scikit-learn joblib streamlit scipy seaborn matplotlib jupyter
```

## Train The Models

Train the freight model:

```bash
python fregit_cost_prediction/train.py
```

Train the invoice-risk classifiers:

```bash
python Invoice_flagging/train.py
```

The best models and metadata are generated in their respective `models/` directories. These generated files are excluded by `.gitignore`.

The risk training pipeline chooses the best classifier by macro F1 score because the risk classes are not evenly distributed.

## Run Predictions

Predict freight cost for the example unseen invoices:

```bash
python predict_freight.py
```

The freight prediction input must contain:

```text
Quantity,Dollars
```

Predict invoice risk for the example unseen invoices:

```bash
python predict_invoice_flagging.py
```

Risk prediction requires these input fields:

```text
invoice_quantity,
invoice_dollars,
total_item_quantity,
total_item_dollars,
days_po_to_invoice,
days_to_pay,
avg_receiving_delay
```

## Run The Web App

Start Streamlit from the project root:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

The dashboard provides:

- A freight forecast form.
- A single-invoice risk assessment form.
- Batch CSV risk scoring.
- A download button for scored invoices.

## Important Notes

- Run the training scripts before using the dashboard if model files do not exist.
- The database must be available at `data/inventory.db` for training.
- The invoice-risk labels are generated from rule-based checks. Models trained on those same engineered features can achieve very high scores because the features closely reflect the labeling rules.
- SQLite databases, model artifacts, Python caches, and notebook checkpoints are ignored by Git.
