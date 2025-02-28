import pandas as pd
import matplotlib.pyplot as plt
import colorsys
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
import matplotlib.colors as mc
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
import numpy as np

import pdb

def convert_celsius_proper_unit(temp):
    kelvin = temp + 273.15
    return 1000/kelvin

def convert_expt_error_log_unit(df):
    true_y = 10**(df['Actual'])
    error = df['Error']
    true_y_lower = true_y - error; log_true_y_lower = np.log10(true_y_lower)
    true_y_upper = true_y + error; log_true_y_upper = np.log10(true_y_upper)
    log_error = log_true_y_upper - log_true_y_lower
    return log_error

df_add = pd.read_csv("/project/rcc/hyadav/TransPolymer_3/TransPolymer/data/expt_valid_add.csv")
df_smiles = pd.read_csv("/project/rcc/hyadav/TransPolymer_3/TransPolymer/data/expt_valid_comb_comm.csv")
# df_preds = pd.read_csv("/project/rcc/hyadav/TransPolymer_3/TransPolymer/plots/inference_plot_rmse_0.22469916829357195_r2_0.13932651281356812_mae_0.16327066719532013.csv") #Rand Model Expt Scaled Expt Nonfusion
# df_preds = pd.read_csv("/project/rcc/hyadav/TransPolymer_3/TransPolymer/plots/inference_plot_rmse_0.23731299962686142_r2_0.03998380899429321_mae_0.1650082916021347.csv") #Strat Model Scaled Expt Nonfusion
# df_preds = pd.read_csv("/project/rcc/hyadav/TransPolymer_2/plots/inference_plot_rmse_0.2698952525103282_r2_-0.24172663688659668_mae_0.17901761829853058.csv") #Strat Fusion
df_preds = pd.read_csv("/project/rcc/hyadav/TransPolymer_2/plots/inference_plot_rmse_0.2519721829798977_r2_-0.0822826623916626_mae_0.1828807294368744.csv") #Rand Fusion
 
combined_df = pd.concat([df_smiles, df_add], axis= 1)
combined_df = pd.concat([combined_df, df_preds], axis= 1)

em = list(combined_df.query("solv_comb_sm == 'CCN1CCOCC1'").index)
mcf = list(combined_df.query("solv_comb_sm == 'COC(=O)C#N'").index)
esf = list(combined_df.query("solv_comb_sm == 'CCS(=O)(=O)F'").index)
eesf = list(combined_df.query("solv_comb_sm == 'CCOCCS(=O)(=O)F'").index)

combined_df['log_error_expt'] = convert_expt_error_log_unit(combined_df)

# #Plot
# fig, ax = plt.subplots(figsize=(6,6))
# # Create a gridspec to specify the layout of the figure
# gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])

# # Create the subplots
# ax1 = plt.subplot(gs[1])
# ax2 = plt.subplot(gs[0])

# # # Remove duplicate tick labels
# # plt.setp(ax1.get_xticklabels(), visible=False)
# # plt.setp(ax2.get_yticklabels(), visible=False)

# uniqs = [em, esf, eesf]
# colors = [(0.984, 0.713, 0.305),(0.298, 0.78, 0.77),(0, 0.576, 0.902)]
# labels = ['EM', 'ESF', 'EESF']
# markers = ['o', 'D', '^']

# x = convert_celsius_proper_unit(combined_df['temperature'][em])

# print(x)

# bar_width = 0.025

# print(combined_df)

# # Loop over each unique value
# for i in range(len(uniqs)):
#     ind = uniqs[i]
    
#     # Get the true and predicted values
#     y_true = combined_df['Actual'][ind]; y_true_ = 10**y_true; y_true_ln = np.log(y_true_)
#     y_pred = combined_df['Predicted'][ind]; y_pred_ = 10**y_pred; y_pred_ln = np.log(y_pred_)
    
#     # Fit a line to the true and predicted values
#     lr_fit_true = np.polyfit(x, y_true, deg=1); lr_fit_true_ln = np.polyfit(x, y_true_ln, deg=1)
#     lr_fit_pred = np.polyfit(x, y_pred, deg=1); lr_fit_pred_ln = np.polyfit(x, y_pred_ln, deg=1)
    
#     # Create line functions
#     lr_fn_true = np.poly1d(lr_fit_true); lr_fn_true_ln = np.poly1d(lr_fit_true_ln)
#     lr_fn_pred = np.poly1d(lr_fit_pred); lr_fn_pred_ln = np.poly1d(lr_fit_pred_ln)
#     act_energy_pred = -lr_fit_pred_ln[0] * 8.314
#     act_energy_true = -lr_fit_true_ln[0] * 8.314
#     print("True & predicted activation energies for", labels[i], ": ", act_energy_true, "& ", act_energy_pred) 
#     # print("True activation energy for", labels[i], ": ", act_energy_true)
    
#     # Plot the true and predicted values as scatter plots
#     # ax1.errorbar(x, y_true, yerr=df_pred['error_conductivity'][ind], marker=markers[i], linewidth=0, elinewidth=1, ecolor=colors[i], markersize=size[i], capsize=3, markeredgewidth=0.5, markeredgecolor=colors[i])
#     # ax1.errorbar(x, y_true, yerr=combined_df['log_error_expt'][ind], marker=markers[i], linewidth=0, elinewidth=1, ecolor=colors[i], markerfacecolor=colors[i], capsize=3, markeredgewidth=0.5, markeredgecolor=colors[i])
#     # # ax1.scatter(x, y_true, color=colors[i], marker=markers[i])
#     # ax1.errorbar(x, y_pred, yerr=combined_df['Predicted'][ind], marker=markers[i], linewidth=0, elinewidth=1, ecolor=colors[i], markerfacecolor='none', capsize=3, markeredgewidth=0.5, markeredgecolor=colors[i])
#     # ax1.scatter(x, y_pred, color=colors[i], marker=markers[i], edgecolors=colors[i], facecolors='none')
    
#     # Plot the fitted lines
#     ax1.plot(x, lr_fn_true(x), linestyle='-', color=colors[i], linewidth=1.0, label=f'{labels[i]} - GT')
#     ax1.plot(x, lr_fn_pred(x), linestyle='--', color=colors[i], linewidth=1.0, label=f'{labels[i]} - Pred')
    
#     # Calculate the absolute errors
#     abs_errors = np.abs(y_true - y_pred)
    
#     # Plot the absolute errors as a bar plot
#     # x_offset = x + bar_width * (i - len(uniqs) / 2)
#     # ax2.bar(x, abs_errors, color=colors[i], alpha=1.0, width=bar_width)
#     ax2.bar(x + i * bar_width, abs_errors, color=colors[i], alpha=0.8, width=bar_width, label=labels[i])


# # Also on bottom of the bottom plot
# # ax1.xaxis.tick_bottom()
# # ax1.set_xticks(x)
# # ax1.set_xticklabels(round(x,3))

# ax1.set_xticks(x)
# ax1.set_xticklabels(round(x, 3), rotation=30)

# # Create a twin axis for the bottom plot
# ax1_twin = ax1.twiny()

# # Set xticks on top of the bottom plot with df_pred['temperature'] labels
# ax1_twin.set_xticks(x)
# ax1_twin.set_xticklabels(combined_df['temperature'].unique(), rotation=30)

# # ax1.set_xlim([2.81, 3.44])
# # ax1_twin.set_xlim([2.81, 3.44])

# # ax2.set_xlim([2.81, 3.44])
# # ax2.set_xticks(x)
# # ax2.set_xticklabels([])


# # Set the labels
# ax1.set_xlabel('1000/T (K)')
# # ax1.set_ylabel('log$_{10}{}\sigma$ (mS/cm)')
# ax1.set_ylabel('log $\sigma$ (mS cm$^{-1}$)')
# ax2.set_ylabel('|y$_{true}$ - y$_{ML}$|')

# # Legends
# # Improved legends placement
# ax1.legend(loc='upper left', fontsize=8, bbox_to_anchor=(1.05, 1.0), borderaxespad=0)
# ax2.legend(loc='upper left', fontsize=8, bbox_to_anchor=(1.05, 1.0), borderaxespad=0)

# # Consistent x-axis limits
# x_margin = 0.05
# ax1.set_xlim([x.min() - x_margin, x.max() + x_margin])
# ax1_twin.set_xlim(ax1.get_xlim())

# ax2.set_xticks(x)
# ax2.set_xticklabels([])

# # Show the plot
# plt.tight_layout()
# plt.subplots_adjust(right=0.8) 
# # plt.show()
# plt.savefig('/project/rcc/hyadav/TransPolymer_3/TransPolymer/expt_plots/expt_rand_2025.png', dpi=300)
# print("Save /project/rcc/hyadav/TransPolymer_3/TransPolymer/expt_plots/expt_rand_2025.png")

###PLOT

# Example data (replace with your actual data)
uniqs = [em, esf, eesf]
colors = [(0.984, 0.713, 0.305), (0.298, 0.78, 0.77), (0, 0.576, 0.902)]
labels = ['EM', 'ESF', 'EESF']
markers = ['o', 'D', '^']

# Example x values
x = convert_celsius_proper_unit(combined_df['temperature'][em])

# Create the plot
fig, ax1 = plt.subplots(figsize=(6, 6))

# Bar width for errors
bar_width = 0.025

# Loop over each unique value to plot
for i in range(len(uniqs)):
    ind = uniqs[i]
    y_true = combined_df['Actual'][ind]
    y_pred = combined_df['Predicted'][ind]

    # Fit linear regression lines
    lr_fit_true = np.polyfit(x, y_true, deg=1)
    lr_fit_pred = np.polyfit(x, y_pred, deg=1)
    lr_fn_true = np.poly1d(lr_fit_true)
    lr_fn_pred = np.poly1d(lr_fit_pred)

    # Plot the fitted lines
    ax1.plot(x, lr_fn_true(x), linestyle='-', color=colors[i], linewidth=1.0, label=f'{labels[i]} - GT')
    ax1.plot(x, lr_fn_pred(x), linestyle='--', color=colors[i], linewidth=1.0, label=f'{labels[i]} - Pred')

# Format the x-axis labels
ax1.set_xticks(x)
ax1.set_xticklabels(np.round(x, 3), rotation=30)

# Create a twin x-axis to add temperature labels
ax1_twin = ax1.twiny()
ax1_twin.set_xticks(x)
ax1_twin.set_xticklabels(df_add['temperature'].iloc[:7])

ax1.set_xlim([2.81, 3.44])
ax1_twin.set_xlim([2.81, 3.44])

# Set the labels
ax1.set_xlabel('1000/T (K)')
ax1.set_ylabel('log $\sigma$ (mS cm$^{-1}$)')

# Add a legend
ax1.legend(loc='upper left', fontsize=6, bbox_to_anchor=(1.05, 1.0), borderaxespad=0)

# Adjust the layout and save the plot
plt.tight_layout()
plt.subplots_adjust(right=0.8)
plt.savefig('/project/rcc/hyadav/TransPolymer_2/expt_plots_fusion/expt_rand_2025.png', dpi=300)
print("Saved: /project/rcc/hyadav/TransPolymer_2/expt_plots_fusion/expt_rand_2025.png")