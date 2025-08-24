import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# ---- 1. Model Definition (MMoE with Temperature Scaling) ----
class Expert(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, p_dropout=0.3):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p_dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_size))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class Tower(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, p_dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        return self.net(x)

class MMoE(nn.Module):
    def __init__(self, input_size, num_experts, expert_hidden, expert_output_dim, tower_hidden_dim, task_output_dims, temperature=1.0):
        super().__init__()
        self.num_tasks = len(task_output_dims)
        self.temperature = nn.Parameter(torch.tensor(temperature), requires_grad=True)
        self.experts = nn.ModuleList(
            Expert(input_size, expert_hidden, expert_output_dim)
            for _ in range(num_experts)
        )
        self.gates = nn.ModuleList(
            nn.Linear(input_size, num_experts, bias=False)
            for _ in range(self.num_tasks)
        )
        self.towers = nn.ModuleList(
            Tower(expert_output_dim, tower_hidden_dim, out_dim)
            for out_dim in task_output_dims
        )
    def forward(self, x):
        expert_stack = torch.stack([expert(x) for expert in self.experts], dim=0)
        outputs = []
        for gate, tower in zip(self.gates, self.towers):
            weights = F.softmax(gate(x), dim=1)
            weighted = torch.einsum("be,ebd->bd", weights, expert_stack)
            y = tower(weighted)
            # Temperature scaling for calibration
            y = y / self.temperature
            outputs.append(y)
        return outputs

# ---- 2. Data Loading ----
col_list = [
    "age", "gender", "n_ed_30d", "n_ed_90d", "n_ed_365d", "n_hosp_30d", "n_hosp_90d", "n_hosp_365d", "n_icu_30d", "n_icu_90d", "n_icu_365d", "triage_temperature", "triage_heartrate", "triage_resprate", "triage_o2sat", "triage_sbp", "triage_dbp", "triage_pain", "triage_acuity", "chiefcom_chest_pain", "chiefcom_abdominal_pain", "chiefcom_headache", "chiefcom_shortness_of_breath", "chiefcom_back_pain", "chiefcom_cough", "chiefcom_nausea_vomiting", "chiefcom_fever_chills", "chiefcom_syncope", "chiefcom_dizziness", "cci_MI", "cci_CHF", "cci_PVD", "cci_Stroke", "cci_Dementia", "cci_Pulmonary", "cci_Rheumatic", "cci_PUD", "cci_Liver1", "cci_DM1", "cci_DM2", "cci_Paralysis", "cci_Renal", "cci_Cancer1", "cci_Liver2", "cci_Cancer2", "cci_HIV", "eci_Arrhythmia", "eci_Valvular", "eci_PHTN",  "eci_HTN1", "eci_HTN2", "eci_NeuroOther", "eci_Hypothyroid", "eci_Lymphoma", "eci_Coagulopathy", "eci_Obesity", "eci_WeightLoss", "eci_FluidsLytes", "eci_BloodLoss", "eci_Anemia", "eci_Alcohol", "eci_Drugs","eci_Psychoses", "eci_Depression"
]
outcome_list = [
    "outcome_hospitalization", "outcome_critical", "outcome_ed_revisit_3d", "outcome_sepsis", "outcome_pneumonia_bacterial", "outcome_pneumonia_viral", "outcome_pneumonia_all", "outcome_ards", "outcome_pe", "outcome_copd_exac", "outcome_acs_mi", "outcome_stroke", "outcome_aki"
]

df = pd.read_csv("../data/master_dataset_with_outcomes.csv")
raw_feats = df.loc[:, col_list]
feat_df = raw_feats.select_dtypes(include=[np.number, bool])
feat_df = feat_df.fillna(feat_df.mean())
X_np = feat_df.to_numpy(dtype=float)
y_np = df.loc[:, outcome_list].to_numpy(dtype=float)

X_t = torch.from_numpy(X_np).float()
y_t = torch.from_numpy(y_np).float()
dataset = TensorDataset(X_t, y_t)

# Use all data for both train and test (as requested)
batch_size = 256
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ---- 3. Training ----
input_size = X_np.shape[1]
num_experts = 8
expert_hidden = [128, 64]
expert_output_dim = 32
tower_hidden_dim = 16
task_output_dims = [1] * len(outcome_list)

model = MMoE(input_size, num_experts, expert_hidden, expert_output_dim, tower_hidden_dim, task_output_dims, temperature=1.0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits_list = model(xb)
        loss = 0
        for i, logits in enumerate(logits_list):
            loss += criterion(logits.squeeze(), yb[:, i])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# ---- 4. Evaluation and Calibration Curve ----
model.eval()
all_probs = []
all_trues = []
with torch.no_grad():
    for xb, yb in dataloader:
        xb = xb.to(device)
        logits_list = model(xb)
        probs = [torch.sigmoid(logits.cpu()).numpy().squeeze() for logits in logits_list]
        all_probs.append(np.stack(probs, axis=1))
        all_trues.append(yb.numpy())
all_probs = np.concatenate(all_probs, axis=0)
all_trues = np.concatenate(all_trues, axis=0)

# Plot calibration curve for each task
plt.figure(figsize=(16, 10))
for i, task in enumerate(outcome_list):
    prob_true, prob_pred = calibration_curve(all_trues[:, i], all_probs[:, i], n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=task)
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration Curves (with Temperature Scaling)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
