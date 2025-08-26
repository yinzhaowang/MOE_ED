# Copy of mmoe_iso_calibration.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve, IsotonicRegression
import matplotlib.pyplot as plt

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- 1. Model Definition (MMoE without Temperature Scaling) ----
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
    def __init__(self, input_size, num_experts, expert_hidden, expert_output_dim, tower_hidden_dim, task_output_dims):
        super().__init__()
        self.num_tasks = len(task_output_dims)
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
            outputs.append(y)
        return outputs

# ---- 2. Placeholder for Isotonic Calibration ----
class IsotonicCalibrator:
    def __init__(self, num_tasks):
        self.num_tasks = num_tasks
        self.calibrators = [IsotonicRegression(out_of_bounds='clip') for _ in range(num_tasks)]

    def fit(self, logits, targets):
        for i in range(self.num_tasks):
            self.calibrators[i].fit(logits[:, i], targets[:, i])

    def predict(self, logits):
        calibrated_probs = np.zeros_like(logits)
        for i in range(self.num_tasks):
            calibrated_probs[:, i] = self.calibrators[i].predict(logits[:, i])
        return calibrated_probs

# ---- 2. Data Loading ----
col_list = [
    "age", "gender", "n_ed_30d", "n_ed_90d", "n_ed_365d", "n_hosp_30d", "n_hosp_90d", "n_hosp_365d", "n_icu_30d", "n_icu_90d", "n_icu_365d", "triage_temperature", "triage_heartrate", "triage_resprate", "triage_o2sat", "triage_sbp", "triage_dbp", "triage_pain", "triage_acuity", "chiefcom_chest_pain", "chiefcom_abdominal_pain", "chiefcom_headache", "chiefcom_shortness_of_breath", "chiefcom_back_pain", "chiefcom_cough", "chiefcom_nausea_vomiting", "chiefcom_fever_chills", "chiefcom_syncope", "chiefcom_dizziness", "cci_MI", "cci_CHF", "cci_PVD", "cci_Stroke", "cci_Dementia", "cci_Pulmonary", "cci_Rheumatic", "cci_PUD", "cci_Liver1", "cci_DM1", "cci_DM2", "cci_Paralysis", "cci_Renal", "cci_Cancer1", "cci_Liver2", "cci_Cancer2", "cci_HIV", "eci_Arrhythmia", "eci_Valvular", "eci_PHTN",  "eci_HTN1", "eci_HTN2", "eci_NeuroOther", "eci_Hypothyroid", "eci_Lymphoma", "eci_Coagulopathy", "eci_Obesity", "eci_WeightLoss", "eci_FluidsLytes", "eci_BloodLoss", "eci_Anemia", "eci_Alcohol", "eci_Drugs","eci_Psychoses", "eci_Depression"
]
outcome_list = [
    "outcome_hospitalization", "outcome_critical", "outcome_ed_revisit_3d", "outcome_sepsis", "outcome_pneumonia_bacterial", "outcome_pneumonia_viral", "outcome_pneumonia_all", "outcome_ards", "outcome_pe", "outcome_copd_exac", "outcome_acs_mi", "outcome_stroke", "outcome_aki"
]


