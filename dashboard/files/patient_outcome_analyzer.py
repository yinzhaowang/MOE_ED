#!/usr/bin/env python3
# Patient Outcome Analyzer - Identifies patients with specific outcome combinations

import pandas as pd
import numpy as np
import os
import sys
from mmoe_iso_calibration import outcome_list

def load_data():
    """Load the test dataframe from the same location as the dashboard app"""
    data_path = "../../../data/test_with_outcomes.csv"
    try:
        df_test = pd.read_csv(data_path)
        print(f"Successfully loaded test data with {len(df_test)} patients.")
        return df_test
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def get_patients_with_sepsis_and_ards_and_other(df):
    """
    Find patients who have sepsis, ARDS, and at least one other outcome
    Returns a list of row indices
    """
    # Create a mask for each condition
    sepsis_mask = df['outcome_sepsis'] == 1
    ards_mask = df['outcome_ards'] == 1
    
    # Calculate the sum of other outcomes excluding sepsis and ards
    other_outcomes = [col for col in outcome_list if col not in ['outcome_sepsis', 'outcome_ards']]
    other_mask = df[other_outcomes].sum(axis=1) > 0
    
    # Combine the masks to find patients with all three conditions
    combined_mask = sepsis_mask & ards_mask & other_mask
    
    # Get the row indices
    patient_indices = df[combined_mask].index.tolist()
    
    return patient_indices

def get_patients_with_only_sepsis_and_ards(df):
    """
    Find patients who have ONLY sepsis and ARDS, but no other outcomes
    Returns a list of row indices
    """
    # Create a mask for each condition
    sepsis_mask = df['outcome_sepsis'] == 1
    ards_mask = df['outcome_ards'] == 1
    
    # Calculate the sum of other outcomes excluding sepsis and ards
    other_outcomes = [col for col in outcome_list if col not in ['outcome_sepsis', 'outcome_ards']]
    no_other_mask = df[other_outcomes].sum(axis=1) == 0
    
    # Combine the masks to find patients with only sepsis and ards
    combined_mask = sepsis_mask & ards_mask & no_other_mask
    
    # Get the row indices
    patient_indices = df[combined_mask].index.tolist()
    
    return patient_indices

def get_patients_with_only_sepsis(df):
    """
    Find patients who have ONLY sepsis, but no other outcomes
    Returns a list of row indices
    """
    # Create a mask for sepsis
    sepsis_mask = df['outcome_sepsis'] == 1
    
    # Calculate the sum of other outcomes excluding sepsis
    other_outcomes = [col for col in outcome_list if col != 'outcome_sepsis']
    no_other_mask = df[other_outcomes].sum(axis=1) == 0
    
    # Combine the masks to find patients with only sepsis
    combined_mask = sepsis_mask & no_other_mask
    
    # Get the row indices
    patient_indices = df[combined_mask].index.tolist()
    
    return patient_indices

def print_patient_summary(df, patient_indices, description):
    """
    Print a summary of the specified patients
    """
    print(f"\n{description} (Total: {len(patient_indices)})")
    print("-" * 80)
    
    if not patient_indices:
        print("No patients found with these criteria.")
        return
    
    print(f"Row indices: {', '.join(map(str, patient_indices[:20]))}")
    if len(patient_indices) > 20:
        print(f"... and {len(patient_indices) - 20} more")
    
    # Print a summary of all outcomes for these patients
    if len(patient_indices) > 0:
        outcome_counts = df.loc[patient_indices, outcome_list].sum().sort_values(ascending=False)
        print("\nOutcome distribution for these patients:")
        for outcome, count in outcome_counts.items():
            if count > 0:
                percentage = (count / len(patient_indices)) * 100
                print(f"  {outcome.replace('outcome_', '')}: {count} patients ({percentage:.1f}%)")

def main():
    """Main function to analyze patient outcomes"""
    print("Patient Outcome Analyzer")
    print("=" * 80)
    
    # Load the data
    df_test = load_data()
    
    # Identify patients with different outcome combinations
    sepsis_ards_other_indices = get_patients_with_sepsis_and_ards_and_other(df_test)
    only_sepsis_ards_indices = get_patients_with_only_sepsis_and_ards(df_test)
    only_sepsis_indices = get_patients_with_only_sepsis(df_test)
    
    # Print summaries
    print_patient_summary(df_test, sepsis_ards_other_indices, 
                          "Patients with Sepsis, ARDS, and at least one other outcome")
    print_patient_summary(df_test, only_sepsis_ards_indices, 
                          "Patients with ONLY Sepsis and ARDS (no other outcomes)")
    print_patient_summary(df_test, only_sepsis_indices, 
                          "Patients with ONLY Sepsis (no other outcomes)")
    
    # Save the results to a CSV file for easy reference
    results = {
        'sepsis_ards_other': sepsis_ards_other_indices,
        'only_sepsis_ards': only_sepsis_ards_indices,
        'only_sepsis': only_sepsis_indices
    }
    
    # Create a dataframe with row indices for each category
    max_len = max(len(v) for v in results.values())
    result_df = pd.DataFrame({
        k: pd.Series(v + [None] * (max_len - len(v))) for k, v in results.items()
    })
    
    # Save to CSV
    output_file = "patient_outcome_groups.csv"
    result_df.to_csv(output_file, index=False)
    print(f"\nSaved results to {output_file}")
    
    # Generate a summary of test cases for the dashboard
    print("\nSample test case rows to use in the dashboard:")
    if sepsis_ards_other_indices:
        print(f"  Patient with Sepsis, ARDS and other: Row {sepsis_ards_other_indices[0]}")
    if only_sepsis_ards_indices:
        print(f"  Patient with only Sepsis and ARDS: Row {only_sepsis_ards_indices[0]}")
    if only_sepsis_indices:
        print(f"  Patient with only Sepsis: Row {only_sepsis_indices[0]}")

if __name__ == "__main__":
    main()
