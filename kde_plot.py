import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pdb

df_rand_train = pd.read_csv("/project/rcc/hyadav/TransPolymer_2/data/new-chemprop-data/train_rand_comb_comm.csv", index_col=None)
df_rand_val = pd.read_csv("/project/rcc/hyadav/TransPolymer_2/data/new-chemprop-data/val_rand_comb_comm.csv", index_col=None)
df_rand_test = pd.read_csv("/project/rcc/hyadav/TransPolymer_2/data/new-chemprop-data/test_rand_comb_comm.csv", index_col=None)

df_clus1_train = pd.read_csv("/project/rcc/hyadav/TransPolymer_2/data/new-chemprop-data/train_clus1_comb_comm.csv", index_col=None)
df_clus1_val = pd.read_csv("/project/rcc/hyadav/TransPolymer_2/data/new-chemprop-data/val_clus1_comb_comm.csv", index_col=None)
df_clus1_test = pd.read_csv("/project/rcc/hyadav/TransPolymer_2/data/new-chemprop-data/test_clus1_comb_comm.csv", index_col=None)

df_clus2_train = pd.read_csv("/project/rcc/hyadav/TransPolymer_2/data/freqII_train_multi_comp_comb.csv", index_col=None)
df_clus2_val = pd.read_csv("/project/rcc/hyadav/TransPolymer_2/data/freqII_val_multi_comp_comb.csv", index_col=None)
df_clus2_test = pd.read_csv("/project/rcc/hyadav/TransPolymer_2/data/freqII_test_multi_comp_comb.csv", index_col=None)
df_clus2_train['conductivity_log'] = np.log10(np.exp(df_clus2_train['conductivity_log']))
df_clus2_val['conductivity_log'] = np.log10(np.exp(df_clus2_val['conductivity_log']))
df_clus2_test['conductivity_log'] = np.log10(np.exp(df_clus2_test['conductivity_log']))

df_strat_train = pd.read_csv("/project/rcc/hyadav/TransPolymer_2/data/new-chemprop-data/train_strat_comp_comm.csv", index_col=None)
df_strat_val =  pd.read_csv("/project/rcc/hyadav/TransPolymer_2/data/new-chemprop-data/val_strat_comp_comm.csv", index_col=None)
df_strat_test = pd.read_csv("/project/rcc/hyadav/TransPolymer_2/data/new-chemprop-data/test_strat_comp_comm.csv", index_col=None)

df_ood = pd.read_csv("/project/rcc/hyadav/TransPolymer_2/data/new-chemprop-data/cond_ood_final_comp.csv", index_col=None)
df_ood['conductivity_log'] = np.log10(np.exp(df_ood['conductivity_log']))

df_ood_best = pd.read_csv("/project/rcc/hyadav/TransPolymer_2/plots/inference_plot_rmse_0.15324774318545523_r2_0.24174129962921143_mae_0.11499364674091339.csv", index_col=None) #OOD for OOD scaled Strat Model delta 6
df_ood_best['conductivity_log'] = df_ood_best['Predicted']

df_ood_chemprop_best = pd.read_csv("/project2/chibueze/riteshk/electrolyte-data/preds_cond_ood_chemprop.csv", index_col=None)

# pdb.set_trace()

datasets = { 
            "Random Train": df_rand_train,
            # "Random Validation": df_rand_val,
            "Random Test": df_rand_test,
            # "Cluster I Train": df_clus1_train,
            # "Cluster I Validation": df_clus1_val,
            "Cluster I Test": df_clus1_test,
            # "Cluster II Train": df_clus2_train,
            # "Cluster II Validation": df_clus2_val,
            "Cluster II Test": df_clus2_test, 
            "Stratified Train": df_strat_train,
            # "Stratified Validation": df_strat_val,
            "Stratified Test": df_strat_test,           
            "OOD": df_ood,
            # "OOD TransPolymer": df_ood_best,
            # "OOD ChemProp": df_ood_chemprop_best
           }

plt.figure(figsize=(6, 5))

palette = sns.color_palette("pastel")


for i, (dataset_name, df) in enumerate(datasets.items()):
    print(dataset_name)
    if dataset_name == "OOD" or dataset_name == "OOD TransPolymer" or dataset_name == "OOD ChemProp" :
        shade_bool = True
    else: 
        shade_bool = False
    sns.kdeplot(df['conductivity_log'], linewidth=3, fill= shade_bool, label = dataset_name, multiple="stack", common_norm=False, color=palette[i % len(palette)])

plt.legend(prop={'size': 12}, title = 'Dataset', fontsize = 12)
plt.title('Kernel Density Estimate Plots', fontsize = 12, fontweight='bold')
plt.xlabel('Ionic Conductivity in log $\sigma$ (log mS cm$^{-1}$)', fontsize = 12)
plt.ylabel('Probability Density', fontsize = 12)
plt.tight_layout
file_name = ("/project/rcc/hyadav/TransPolymer_2/figs/kde_plots/density_plot_all.png")
plt.savefig(file_name)