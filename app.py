from pathlib import Path

import pandas as pd
import streamlit as st

from predict_freight import predict_freight
from predict_invoice_flagging import predict_invoice_risk


st.set_page_config(
	page_title="Invoice Intelligence",
	page_icon="📊",
	layout="wide",
	initial_sidebar_state="expanded",
)


ROOT = Path(__file__).resolve().parent
FREIGHT_MODEL_PATH = ROOT / "fregit_cost_prediction" / "models" / "best_model.pkl"
RISK_MODEL_PATH = ROOT / "Invoice_flagging" / "models" / "best_model.joblib"


st.markdown(
	"""
	<style>
	.stApp { background: #f7f8fa; }
	[data-testid="stSidebar"] { background: #16212b; }
	[data-testid="stSidebar"] * { color: #eef4f7; }
	.hero {
		background: linear-gradient(135deg, #123c4a 0%, #1f6870 100%);
		padding: 2rem 2.25rem;
		border-radius: 14px;
		color: white;
		margin-bottom: 1.5rem;
	}
	.hero h1 { margin: 0; font-size: 2.4rem; }
	.hero p { margin: .5rem 0 0; color: #d8eff0; }
	.result-card {
		background: white;
		border: 1px solid #dce4e8;
		border-radius: 12px;
		padding: 1.25rem;
		box-shadow: 0 5px 18px rgba(18, 60, 74, .06);
	}
	</style>
	""",
	unsafe_allow_html=True,
)


def status_row(label: str, path: Path) -> None:
	state = "Available" if path.is_file() else "Missing"
	st.sidebar.write(f"**{label}:** {state}")


st.sidebar.title("Invoice Intelligence")
st.sidebar.caption("Prediction workspace")
st.sidebar.divider()
st.sidebar.subheader("Model status")
status_row("Freight model", FREIGHT_MODEL_PATH)
status_row("Risk model", RISK_MODEL_PATH)
st.sidebar.divider()
st.sidebar.caption("Models are loaded from the local project model directories.")

st.markdown(
	"""
	<div class="hero">
		<h1>Invoice Intelligence</h1>
		<p>Estimate freight costs and surface invoices that deserve a closer look.</p>
	</div>
	""",
	unsafe_allow_html=True,
)


freight_tab, risk_tab = st.tabs(["Freight forecast", "Invoice risk review"])


with freight_tab:
	st.subheader("Freight forecast")
	st.write("Enter the expected quantity and invoice value to estimate freight cost.")

	with st.form("freight_form"):
		quantity = st.number_input(
			"Quantity",
			min_value=0.0,
			value=125.0,
			step=1.0,
		)
		dollars = st.number_input(
			"Invoice dollars",
			min_value=0.0,
			value=22000.0,
			step=100.0,
		)
		submitted = st.form_submit_button("Predict freight", type="primary")

	if submitted:
		if not FREIGHT_MODEL_PATH.is_file():
			st.error("The freight model has not been generated yet.")
		else:
			try:
				input_data = pd.DataFrame(
					{"Quantity": [quantity], "Dollars": [dollars]}
				)
				result = predict_freight(input_data)
				prediction = result.loc[0, "Predicted_Freight"]
				st.success(f"Estimated freight cost: {prediction:,.2f}")
				st.dataframe(result, use_container_width=True, hide_index=True)
			except Exception as error:
				st.error(f"Prediction failed: {error}")


with risk_tab:
	st.subheader("Invoice risk review")
	st.write("Compare invoice values with purchase totals and timing signals.")

	with st.form("risk_form"):
		left, right = st.columns(2)
		with left:
			invoice_quantity = st.number_input(
				"Invoice quantity", min_value=0.0, value=1000.0, step=1.0
			)
			invoice_dollars = st.number_input(
				"Invoice dollars", min_value=0.0, value=12000.0, step=100.0
			)
			total_item_quantity = st.number_input(
				"Purchase quantity", min_value=0.0, value=1000.0, step=1.0
			)
			total_item_dollars = st.number_input(
				"Purchase dollars", min_value=0.0, value=12000.0, step=100.0
			)
		with right:
			days_po_to_invoice = st.number_input(
				"Days from PO to invoice", min_value=0.0, value=10.0, step=1.0
			)
			days_to_pay = st.number_input(
				"Days to pay", min_value=0.0, value=20.0, step=1.0
			)
			avg_receiving_delay = st.number_input(
				"Average receiving delay", min_value=0.0, value=8.0, step=0.5
			)
		risk_submitted = st.form_submit_button("Assess invoice risk", type="primary")

	if risk_submitted:
		if not RISK_MODEL_PATH.is_file():
			st.error("The invoice-risk model has not been generated yet.")
		else:
			try:
				input_data = pd.DataFrame(
					{
						"invoice_quantity": [invoice_quantity],
						"invoice_dollars": [invoice_dollars],
						"total_item_quantity": [total_item_quantity],
						"total_item_dollars": [total_item_dollars],
						"days_po_to_invoice": [days_po_to_invoice],
						"days_to_pay": [days_to_pay],
						"avg_receiving_delay": [avg_receiving_delay],
					}
				)
				result = predict_invoice_risk(input_data)
				risk = result.loc[0, "Predicted_Risk"]
				if risk == "High":
					st.error(f"Risk level: {risk}")
				elif risk == "Medium":
					st.warning(f"Risk level: {risk}")
				else:
					st.success(f"Risk level: {risk}")
				st.dataframe(result, use_container_width=True, hide_index=True)
			except Exception as error:
				st.error(f"Risk assessment failed: {error}")

	st.divider()
	st.subheader("Batch risk scoring")
	st.caption("Upload a CSV containing the seven fields used by the risk model.")
	uploaded_file = st.file_uploader("Upload invoice data", type=["csv"])

	if uploaded_file is not None:
		try:
			uploaded_data = pd.read_csv(uploaded_file)
			scored_data = predict_invoice_risk(uploaded_data)
			st.dataframe(scored_data, use_container_width=True, hide_index=True)
			st.download_button(
				"Download scored invoices",
				data=scored_data.to_csv(index=False).encode("utf-8"),
				file_name="scored_invoices.csv",
				mime="text/csv",
			)
		except Exception as error:
			st.error(f"Batch scoring failed: {error}")
