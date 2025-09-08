import torch
import shap
import numpy as np
import os
import matplotlib.pyplot as plt
from mmoe_iso_calibration import MMoE, Expert, Tower
import pandas as pd

# Load your data (adjust path as needed)
df = pd.read_csv("../data/master_dataset_with_outcomes.csv")
col_list = [
    "age", "gender", "n_ed_30d", "n_ed_90d", "n_ed_365d", "n_hosp_30d", "n_hosp_90d", "n_hosp_365d", "n_icu_30d", "n_icu_90d", "n_icu_365d", "triage_temperature", "triage_heartrate", "triage_resprate", "triage_o2sat", "triage_sbp", "triage_dbp", "triage_pain", "triage_acuity", "chiefcom_chest_pain", "chiefcom_abdominal_pain", "chiefcom_headache", "chiefcom_shortness_of_breath", "chiefcom_back_pain", "chiefcom_cough", "chiefcom_nausea_vomiting", "chiefcom_fever_chills", "chiefcom_syncope", "chiefcom_dizziness", "cci_MI", "cci_CHF", "cci_PVD", "cci_Stroke", "cci_Dementia", "cci_Pulmonary", "cci_Rheumatic", "cci_PUD", "cci_Liver1", "cci_DM1", "cci_DM2", "cci_Paralysis", "cci_Renal", "cci_Cancer1", "cci_Liver2", "cci_Cancer2", "cci_HIV", "eci_Arrhythmia", "eci_Valvular", "eci_PHTN",  "eci_HTN1", "eci_HTN2", "eci_NeuroOther", "eci_Hypothyroid", "eci_Lymphoma", "eci_Coagulopathy", "eci_Obesity", "eci_WeightLoss", "eci_FluidsLytes", "eci_BloodLoss", "eci_Anemia", "eci_Alcohol", "eci_Drugs","eci_Psychoses", "eci_Depression"
]

X_np = df.loc[:, col_list].select_dtypes(include=[np.number, bool]).fillna(0).to_numpy(dtype=float)

# Model setup (match your training config)
input_size = X_np.shape[1]
num_experts = 8
expert_hidden = [128, 64]
expert_output_dim = 32
tower_hidden_dim = 16
task_output_dims = [1] * 13  # Adjust if needed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MMoE(input_size, num_experts, expert_hidden, expert_output_dim, tower_hidden_dim, task_output_dims).to(device)
model.load_state_dict(torch.load("../best_mmoe.pt", map_location=device))
model.eval()

# Use a small background set for SHAP
background = torch.from_numpy(X_np[:100]).float().to(device)
test_samples = torch.from_numpy(X_np[100:110]).float().to(device)

# --- Helper: Wrapper for expert/gate ---
class ExpertWrapper(torch.nn.Module):
    def __init__(self, expert):
        super().__init__()
        self.expert = expert
    def forward(self, x):
        return self.expert(x)

class GateWrapper(torch.nn.Module):
    def __init__(self, gate):
        super().__init__()
        self.gate = gate
    def forward(self, x):
        return torch.softmax(self.gate(x), dim=1)

# --- SHAP for Experts ---
print("\nSHAP for Experts:")
for i, expert in enumerate(model.experts):
    expert_dir = f"SHAP_Plot/expert_{i+1}"
    os.makedirs(expert_dir, exist_ok=True)
    wrapper = ExpertWrapper(expert)
    explainer = shap.DeepExplainer(wrapper, background)
    shap_values = explainer.shap_values(test_samples, check_additivity=False)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    # shap_values shape: (batch, output_dim) or (batch, output_dim, input_dim)
    # We want to aggregate over output_dim if present, to get (batch, input_dim)
    if shap_values.ndim == 3:
        # (batch, output_dim, input_dim) -> (batch, input_dim)
        shap_values_input = np.mean(np.abs(shap_values), axis=1)
    else:
        shap_values_input = shap_values  # (batch, input_dim)
    mean_abs_shap = np.abs(shap_values_input).mean(axis=0)
    n_feat = min(20, mean_abs_shap.shape[0], X_np.shape[1])
    top_idx = np.argsort(mean_abs_shap)[-n_feat:][::-1]
    feature_names = np.array(col_list)[top_idx]
    shap.plots.beeswarm(shap.Explanation(
        values=shap_values_input[:, top_idx],
        data=X_np[100:110][:, top_idx],
        feature_names=feature_names
    ), show=False)
    plt.title(f"Expert {i+1} Top {n_feat} Features (Beeswarm)")
    plt.tight_layout()
    plt.savefig(os.path.join(expert_dir, f"expert_{i+1}_shap_beeswarm.png"))
    plt.close()
    print(f"Expert {i+1} SHAP beeswarm plot saved to {expert_dir}")

# --- SHAP for Gates ---
print("\nSHAP for Gates:")
for t, gate in enumerate(model.gates):
    gate_dir = f"SHAP_Plot/gate_{t+1}"
    os.makedirs(gate_dir, exist_ok=True)
    wrapper = GateWrapper(gate)
    explainer = shap.DeepExplainer(wrapper, background)
    shap_values = explainer.shap_values(test_samples, check_additivity=False)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    # shap_values shape: (batch, output_dim) or (batch, output_dim, input_dim)
    if shap_values.ndim == 3:
        shap_values_input = np.mean(np.abs(shap_values), axis=1)
    else:
        shap_values_input = shap_values
    mean_abs_shap = np.abs(shap_values_input).mean(axis=0)
    n_feat = min(20, mean_abs_shap.shape[0], X_np.shape[1])
    top_idx = np.argsort(mean_abs_shap)[-n_feat:][::-1]
    feature_names = np.array(col_list)[top_idx]
    shap.plots.beeswarm(shap.Explanation(
        values=shap_values_input[:, top_idx],
        data=X_np[100:110][:, top_idx],
        feature_names=feature_names
    ), show=False)
    plt.title(f"Gate {t+1} Top {n_feat} Features (Beeswarm)")
    plt.tight_layout()
    plt.savefig(os.path.join(gate_dir, f"gate_{t+1}_shap_beeswarm.png"))
    plt.close()
    print(f"Gate {t+1} SHAP beeswarm plot saved to {gate_dir}")
