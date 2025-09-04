# Import necessary libraries
from dash import Dash, html, dcc, Input, Output
import pandas as pd
import plotly.express as px
import torch
from torch.utils.data import TensorDataset, DataLoader
from files.mmoe_iso_calibration import MMoE, IsotonicCalibrator, outcome_list, col_list
import numpy as np
import joblib
import os
import math

# Initialize the Dash app
app = Dash(__name__)

# ---- Data Loading ----
df_test = pd.read_csv("../../data/test_with_outcomes.csv")
raw_feats = df_test.loc[:, col_list]
feat_df = raw_feats.select_dtypes(include=[np.number, bool])
feat_df = feat_df.fillna(feat_df.mean())
X_np = feat_df.to_numpy(dtype=float)
y_np = df_test.loc[:, outcome_list].to_numpy(dtype=float)

X_t = torch.from_numpy(X_np).float()
y_t = torch.from_numpy(y_np).float()
dataset = TensorDataset(X_t, y_t)

batch_size = 256  # Define batch size if not already defined
test_dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# ---- Model Loading ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = X_np.shape[1]
num_experts = 8
expert_hidden = [128, 64]
expert_output_dim = 32
tower_hidden_dim = 16
task_output_dims = [1] * len(outcome_list)

model = MMoE(input_size, num_experts, expert_hidden, expert_output_dim, tower_hidden_dim, task_output_dims).to(device)
model.load_state_dict(torch.load("files/best_mmoe_iso.pt", map_location=device))
model.eval()

# ---- Calibration Setup ----
# Load the pre-fitted calibrator
calibrator_path = "files/calibrator.pkl"
if os.path.exists(calibrator_path):
    calibrator = joblib.load(calibrator_path)
    print("Loaded pre-fitted calibrator from file.")
else:
    raise FileNotFoundError(f"Calibrator file not found at {calibrator_path}. Please ensure the file exists.")

# Callback to dynamically calibrate and predict for the selected row
@app.callback(
    [Output("lift-bar-chart", "figure"),
     Output("calibrated-probability-bar-chart", "figure")],
    [Input("row-input", "value")]
)
def update_charts(selected_row):
    # Ensure the selected row is valid
    if selected_row is None or selected_row < 0 or selected_row >= X_np.shape[0]:
        selected_row = 0

    # Extract the selected row
    X_selected = torch.from_numpy(X_np[selected_row:selected_row + 1]).float().to(device)

    # Run the model on the selected row
    with torch.no_grad():
        logits_list = model(X_selected)
        logits = torch.cat(logits_list, dim=1).cpu().numpy()

    # Perform calibration
    calibrated_probs = calibrator.predict(logits)

    # Calculate probability ratios using actual proportions of 1s in each outcome
    train_proportions = df_test[outcome_list].mean().values  # Proportion of 1s for each outcome
    probability_ratios = np.log(calibrated_probs / train_proportions)

    # Create DataFrame
    data = {
        "lift": probability_ratios.flatten(),
        "calibrated probability": calibrated_probs.flatten(),
        "populational baseline": train_proportions
    }
    result_df = pd.DataFrame(data, index=outcome_list)

    # Sort by value descending for each plot
    lift_sorted = result_df.sort_values("lift", ascending=False)
    calibrated_prob_sorted = result_df.sort_values("calibrated probability", ascending=False)

    # Create vertical bar charts (x=values, y=outcomes)
    lift_fig = px.bar(lift_sorted, x="lift", y=lift_sorted.index, orientation="h", title="Lift by Outcome (Descending)")
    calibrated_prob_fig = px.bar(calibrated_prob_sorted, x="calibrated probability", y=calibrated_prob_sorted.index, orientation="h", title="Calibrated Probability by Outcome (Descending)")

    return lift_fig, calibrated_prob_fig

# ---- Dashboard Layout ----
app.layout = html.Div([
    html.H1("ED Decision-Making Dashboard", style={"textAlign": "center"}),

    html.Div([
        html.Label("Select Test Case Row:"),
        dcc.Input(id="row-input", type="number", min=0, max=X_np.shape[0] - 1, step=1, value=0),
    ], style={"margin": "20px"}),

    html.Div([
        html.H2("Selected Test Case Results"),
        dcc.Graph(id="lift-bar-chart"),
        dcc.Graph(id="calibrated-probability-bar-chart"),
    ])
])

# Run the app
if __name__ == "__main__":
    app.run(debug=True)