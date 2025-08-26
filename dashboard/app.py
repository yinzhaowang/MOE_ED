# Import necessary libraries
from dash import Dash, html, dcc, Input, Output
import pandas as pd
import plotly.express as px
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from files.mmoe_iso_calibration import MMoE, IsotonicCalibrator, outcome_list, col_list
import numpy as np

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
test_dl = DataLoader(dataset, batch_size=256, shuffle=False)

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

# ---- Perform Calibration ----
calibrator = IsotonicCalibrator(num_tasks=len(outcome_list))
test_logits = []
with torch.no_grad():
    for X_batch, _ in test_dl:
        X_batch = X_batch.to(device)
        logits_list = model(X_batch)
        logits = torch.cat(logits_list, dim=1)
        test_logits.append(logits)
test_logits = torch.cat(test_logits, dim=0).cpu().numpy()
test_probs = calibrator.predict(test_logits)

# Calculate train proportions (placeholder for actual proportions)
train_proportions = np.random.rand(len(outcome_list))  # Replace with actual proportions
probability_ratios = test_probs / train_proportions

# ---- Dashboard Layout ----
app.layout = html.Div([
    html.H1("ED Decision-Making Dashboard", style={"textAlign": "center"}),

    html.Div([
        html.Label("Select Test Case Row:"),
        dcc.Input(id="row-input", type="number", min=0, max=test_probs.shape[0] - 1, step=1, value=0),
    ], style={"margin": "20px"}),

    html.Div([
        html.H2("Selected Test Case Results"),
        dcc.Graph(id="lift-bar-chart"),
        dcc.Graph(id="calibrated-probability-bar-chart"),
    ])
])

# ---- Callbacks ----
@app.callback(
    [Output("lift-bar-chart", "figure"),
     Output("calibrated-probability-bar-chart", "figure")],
    [Input("row-input", "value")]
)
def update_charts(selected_row):
    # Ensure the selected row is valid
    if selected_row is None or selected_row < 0 or selected_row >= test_probs.shape[0]:
        selected_row = 0

    # Extract data for the selected row
    random_probabilities = test_probs[selected_row, :]
    random_ratios = probability_ratios[selected_row, :]

    # Create DataFrame
    data = {
        "lift": random_ratios,
        "calibrated probability": random_probabilities,
        "populational baseline": train_proportions
    }
    result_df = pd.DataFrame(data, index=outcome_list)

    # Create figures
    lift_fig = px.bar(result_df, x=result_df.index, y="lift", title="Lift by Outcome")
    calibrated_prob_fig = px.bar(result_df, x=result_df.index, y="calibrated probability", title="Calibrated Probability by Outcome")

    return lift_fig, calibrated_prob_fig

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
