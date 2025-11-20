# Import necessary libraries
from dash import Dash, html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
df_test = pd.read_csv("data/test_with_outcomes.csv")
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


def build_preset_cases(num_rows):
    """Generate preset demo cases spread across the dataset."""
    if num_rows <= 0:
        return []
    anchors = sorted(set([
        0,
        max(0, num_rows // 4),
        max(0, num_rows // 2),
        max(0, num_rows - 1)
    ]))
    labels = [
        "Case A – Baseline Triage",
        "Case B – Moderate Risk Signals",
        "Case C – Elevated Complications",
        "Case D – Critical Thresholds"
    ]
    options = []
    for i, idx in enumerate(anchors):
        label = labels[i] if i < len(labels) else f"Case {i + 1}"
        options.append({"label": f"{label} (row {idx})", "value": idx})
    return options


PRESET_CASES = build_preset_cases(X_np.shape[0])

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
     Output("outcome-table", "figure")],
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

    # Remove 'outcome_hospitalization' and clean up variable names
    exclude = 'outcome_hospitalization'
    filtered_outcomes = [o for o in outcome_list if o != exclude]
    calibrated_probs_filtered = calibrated_probs[:, [i for i, o in enumerate(outcome_list) if o != exclude]]
    
    # Use predefined thresholds instead of train_proportions
    thresholds = {
        'outcome_copd_exac': 0.022,
        'outcome_acs_mi': 0.018,
        'outcome_sepsis': 0.022,
        'outcome_critical': 0.080,
        'outcome_stroke': 0.049,
        'outcome_aki': 0.055,
        'outcome_pneumonia_bacterial': 0.003,
        'outcome_ards': 0.001,
        'outcome_hospitalization': 0.471,
        'outcome_pneumonia_viral': 0.002,
        'outcome_pe': 0.010,
        'outcome_pneumonia_all': 0.008,
        'outcome_ed_revisit_3d': 0.030
    }
    
    # Get thresholds for filtered outcomes in the same order
    threshold_values = np.array([thresholds[o] for o in filtered_outcomes])
    
    # Calculate probability ratios using log-odds ratio:
    # log(calibrated/(1-calibrated)) - log(threshold/(1-threshold))
    # This compares the log-odds of the prediction to the log-odds of the threshold
    #calibrated_odds = calibrated_probs_filtered / (1 - calibrated_probs_filtered)
    #threshold_odds = threshold_values / (1 - threshold_values)
    #probability_ratios = np.log(calibrated_odds) - np.log(threshold_odds)
    probability_ratios = np.log(calibrated_probs_filtered/threshold_values)
    # Clean variable names: remove 'outcome_' and capitalize
    def pretty_name(name):
        # Map outcome names to full medical terms
        name_mapping = {
            'outcome_critical': 'Critical Care Admission',
            'outcome_ed_revisit_3d': 'ED Revisit (3 days)',
            'outcome_sepsis': 'Sepsis',
            'outcome_pneumonia_bacterial': 'Bacterial Pneumonia',
            'outcome_pneumonia_viral': 'Viral Pneumonia',
            'outcome_pneumonia_all': 'All-cause Pneumonia',
            'outcome_ards': 'Acute Respiratory Distress Syndrome',
            'outcome_pe': 'Pulmonary Embolism',
            'outcome_copd_exac': 'COPD Exacerbation',
            'outcome_acs_mi': 'Acute Coronary Syndrome/Myocardial Infarction',
            'outcome_stroke': 'Stroke',
            'outcome_aki': 'Acute Kidney Injury'
        }
        
        if name in name_mapping:
            return name_mapping[name]
        else:
            # Fallback in case the name isn't in the mapping
            return name.replace('outcome_', '').replace('_', ' ').capitalize()
    pretty_names = [pretty_name(o) for o in filtered_outcomes]

    # Create DataFrame
    data = {
        "lift": probability_ratios.flatten(),
        "calibrated probability": calibrated_probs_filtered.flatten(),
        "threshold": threshold_values,
        "Outcome": pretty_names
    }
    result_df = pd.DataFrame(data)

    # Sort by lift: ascending order
    lift_sorted = result_df.sort_values("lift", ascending=True)

    # Color for lift: blue for negative, red for positive, with descriptive labels
    def risk_label(x):
        return "Positive risk" if x >= 0 else "Negative risk"
    lift_sorted["risk"] = lift_sorted["lift"].apply(risk_label)

    # Make variable names more visible (larger font, Outcome axis), extend y-axis
    lift_fig = px.bar(
        lift_sorted,
        x="lift",
        y="Outcome",
        orientation="h",
        color="risk",
        color_discrete_map={"Positive risk": "#d62728", "Negative risk": "#1f77b4"},
        title="Risk by Outcome (Log-Lift Ratio, Ascending)",
        labels={"lift": "Log-Lift Ratio", "Outcome": "Outcome", "risk": "Risk Direction"}
    )
    lift_fig.update_layout(
        yaxis=dict(tickfont=dict(size=14), automargin=True),  # Reduced font size from 18 to 16
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),
        height=len(lift_sorted)*30+190,  # Extended height by 40 pixels
        margin=dict(l=0, r=0, t=50, b=50),  # Reduce margins
        xaxis=dict(range=[-4, 4])  # Expanded x-axis range from -2,2 to -4,4
    )
    lift_fig.update_traces(marker_line_width=1.5)

    # Create a table showing actual outcomes for the selected patient
    actual_outcomes = y_np[selected_row]
    actual_outcomes_filtered = [actual_outcomes[i] for i, o in enumerate(outcome_list) if o != exclude]
    
    # Create data for the table
    table_data = {
        "Condition": pretty_names,
        "Predicted Probability (%)": [f"{p*100:.1f}%" for p in calibrated_probs_filtered.flatten()],
        "Actual Outcome": ["Yes" if o == 1 else "No" for o in actual_outcomes_filtered]
    }
    
    # Create a DataFrame and sort it to match the lift plot order
    table_df = pd.DataFrame(table_data)
    
    # IMPORTANT: Make sure to explicitly sort the table in the EXACT same order as the plot
    # But REVERSED to match visual alignment (top to bottom vs bottom to top)
    table_df = pd.DataFrame(table_data)
    # Get the exact order from lift_sorted, but reverse it
    sorted_conditions = lift_sorted['Outcome'].tolist()[::-1]  # Reverse the order
    # Create a new DataFrame preserving this exact order
    table_df = pd.DataFrame({
        'Condition': sorted_conditions,
        'Predicted Probability (%)': [table_data['Predicted Probability (%)'][pretty_names.index(cond)] for cond in sorted_conditions],
        'Actual Outcome': [table_data['Actual Outcome'][pretty_names.index(cond)] for cond in sorted_conditions]
    })
    
    # Create the table figure
    table_fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(table_df.columns),
            fill_color='paleturquoise',
            align='left',
            font=dict(size=14),
            height=25  # Explicitly set header height
        ),
        cells=dict(
            values=[table_df[col] for col in table_df.columns],
            fill_color='lavender',
            align='left',
            font=dict(size=12),
            height=25  # Make all cells same height as lift bars
        ),
        columnwidth=[0.9, 0.6, 0.6]  # Adjusted column widths after removing columns
    )])
    
    table_fig.update_layout(
        title="Patient Outcomes",
        height=len(lift_sorted)*30+190,  # Match the extended lift figure height (+40)
        width=400,  # Narrower table after removing columns
        margin=dict(l=0, r=0, t=50, b=50)  # Reduce margins to match chart
    )
    
    return lift_fig, table_fig  # Return both figures

# ---- Demo View State Management ----
@app.callback(
    Output("demo-mode", "data"),
    Input("launch-demo-button", "n_clicks"),
    State("demo-mode", "data"),
    prevent_initial_call=True
)
def activate_demo(n_clicks, current_state):
    """Flip the demo flag once the user clicks the launch button."""
    if n_clicks:
        return True
    return current_state


@app.callback(
    Output("landing-view", "style"),
    Output("dashboard-view", "style"),
    Input("demo-mode", "data")
)
def toggle_views(demo_active):
    """Hide the landing page and reveal the dashboard once demo mode is active."""
    if demo_active:
        return {"display": "none"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}


@app.callback(
    Output("row-input", "value"),
    Input("preset-selector", "value"),
    prevent_initial_call=True
)
def load_preset_row(selected_row):
    """Populate the row selector when a preset case is chosen."""
    if selected_row is None:
        raise PreventUpdate
    return selected_row

# ---- Dashboard Layout ----
app.layout = html.Div([
    dcc.Store(id="demo-mode", data=False),

    html.Div([
        html.H1("ED Risk Intelligence Demo", style={"textAlign": "center", "marginBottom": "10px"}),
        html.P(
            "Explore how multitask models surface critical ED risks in seconds. "
            "Launch the interactive demo to inspect calibrated probabilities, lift scores, "
            "and actual patient outcomes—similar to the KNOWNET showcase.",
            style={"maxWidth": "800px", "margin": "0 auto", "textAlign": "center"}
        ),
        html.Div([
            html.Button("Launch Demo", id="launch-demo-button", n_clicks=0,
                        style={"padding": "12px 30px", "fontSize": "18px", "cursor": "pointer"})
        ], style={"textAlign": "center", "marginTop": "30px"}),
        html.Div([
            html.Div([
                html.H3("What you'll see"),
                html.Ul([
                    html.Li("Condition-level risk lift ranked from high to low"),
                    html.Li("Calibrated probabilities compared against triage thresholds"),
                    html.Li("Actual patient outcomes for rapid validation")
                ])
            ], style={"maxWidth": "500px", "margin": "40px auto", "textAlign": "left"})
        ])
    ], id="landing-view", style={"padding": "60px 20px"}),

    html.Div([
        html.H1("ED Decision-Making Dashboard", style={"textAlign": "center"}),

        html.Div([
            html.Label("Select Test Case Row:"),
            dcc.Input(id="row-input", type="number", min=0, max=X_np.shape[0] - 1, step=1, value=0),
            dcc.Dropdown(
                id="preset-selector",
                options=PRESET_CASES,
                placeholder="Or choose a curated demo patient",
                clearable=True,
                style={"width": "420px", "marginTop": "10px"}
            )
        ], style={"margin": "20px"}),

        html.Div([
            html.H2("Selected Test Case Results"),
            html.Div([
                html.Div([
                    dcc.Graph(id="lift-bar-chart", config={'displayModeBar': False})
                ], style={"width": "70%", "display": "inline-block", "vertical-align": "top", "padding": "0px"}),
                html.Div([
                    dcc.Graph(id="outcome-table", config={'displayModeBar': False})
                ], style={"width": "30%", "display": "inline-block", "vertical-align": "top", "padding": "0px"})
            ], style={"display": "flex", "align-items": "stretch", "height": "100%"})
        ])
    ], id="dashboard-view", style={"display": "none"})
])

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
