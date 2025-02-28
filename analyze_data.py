import pandas as pd
import numpy as np
import pdb

smiles_tokens = pd.read_csv("/project/rcc/hyadav/TransPolymer_2/data/new-chemprop-data/test_random_exact_all_scaled_fusion_no_mol.csv", index_col = False)
num_data = pd.read_csv("/project/rcc/hyadav/TransPolymer_2/data/new-chemprop-data/test_rand_add.csv")
smiles_data = pd.read_csv("/project/rcc/hyadav/TransPolymer_2/data/new-chemprop-data/test_rand_comp_comm.csv")
smiles_data.fillna('NAN_SMILES', inplace=True)

df = pd.concat([smiles_data, num_data], axis = 1)
# Retain original index
df = df.reset_index()

# Drop the specified columns
columns_to_remove = ["mol_wt_solv_1", "mol_wt_solv_2", "mol_wt_solv_3", "mol_wt_solv_4","mol_wt_salt"]
df = df.drop(columns=columns_to_remove)

filtered_df = df[
    (df['solv_2_sm'] == 'NAN_SMILES') &
    (df['solv_3_sm'] == 'NAN_SMILES') &
    (df['solv_4_sm'] == 'NAN_SMILES')
]

predictions_df = pd.read_csv("/project/rcc/hyadav/TransPolymer_2/final_plots/rand_rand_model_multi.csv")
predictions_df['index'] = predictions_df.index

merged_df = pd.merge(filtered_df, predictions_df, on='index', how='inner')

error_threshold = 0.11 
low_error_df = merged_df[merged_df['Error'] <= error_threshold]

grouped = low_error_df.groupby(['solv_1_sm', 'salt_sm', "temperature"])
# grouped = low_error_df.groupby(['solv_1_sm', 'salt_sm', "conc_salt"])

valid_groups = []

for group_key, group_data in grouped:
    # Check if all columns except `temperature` are constant
    # Check if the group has at least 2 rows
    if len(group_data) >= 2:
        valid_groups.append(group_data)

# Combine all valid groups into a single DataFrame
if valid_groups:
    combined_groups = pd.concat(valid_groups)
    # Save to CSV
    combined_groups.to_csv("valid_groups_salt_conc_variable_error_threshold_0_11.csv", index=False)
    print("All valid groups have been saved.")
else:
    print("No valid groups found with 2 or more elements.")
