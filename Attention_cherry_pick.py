import pandas as pd
import numpy as np
import pdb


results = pd.read_csv('/project/rcc/hyadav/TransPolymer_3/TransPolymer/plots/inference_random_test.csv',index_col=False)
input = pd.read_csv('/project/rcc/hyadav/TransPolymer_3/TransPolymer/data/random_test_exact_all_scaled_nonfusion_no_mol_new.csv')

df = pd.concat([input, results], axis = 1)

keyword = 'COCCOC'
filtered_df = df[df['input'].str.contains(keyword, case=False, na=False)]

print("Number of", keyword, "Rows:", filtered_df.shape[0])

filtered_df_sorted = filtered_df.sort_values(by='Error')
best_predictions = filtered_df_sorted.head(5)
worst_predictions = filtered_df_sorted.tail(5)

print("Best Predictions", best_predictions)
print("Worst Predictions", worst_predictions)
