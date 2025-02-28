import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Arial'
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import mean_absolute_error, mean_squared_error , r2_score
from sklearn.decomposition import PCA
import os, sys, glob
import numpy as np

def plot_jointplot(df, color, filename = None):
    y_true = df['Actual']
    y_pred = df['Predicted']
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    axmin = min(min(y_true), min(y_pred)) - 0.1*(max(y_true)-min(y_true))
    axmax = max(max(y_true), max(y_pred)) + 0.1*(max(y_true)-min(y_true))

    g = sns.jointplot(x=y_true, y=y_pred, kind='reg', color=color, xlim=(axmin, axmax), ylim=(axmin, axmax), marginal_kws=dict(kde=True, fill=True), scatter_kws={'edgecolor': 'w', 'linewidths': 0.5})
    g.ax_joint.set_xlim(axmin, axmax)
    g.ax_joint.set_ylim(axmin, axmax)
    x_space = 0.15 * axmax
    y_space = 0.1 * axmax
    plt.text(axmin+x_space, axmax-y_space, 'MAE: {:.2f}'.format(mae), fontsize=12, color=color)
    plt.text(axmin+x_space, axmax-2.5*y_space, 'RMSE: {:.2f}'.format(rmse), fontsize=12, color=color)
    plt.text(axmin+x_space, axmax-4*y_space, 'R$^2$: {:.2f}'.format(r2), fontsize=12, color=color)
    plt.ylabel('Predicted log $\sigma$ (log mS cm$^{-1}$)', fontdict={'fontsize': 14})
    plt.xlabel('True log $\sigma$ (log mS cm$^{-1}$)', fontdict={'fontsize': 14})
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.title(f'Jointplot for {variable_name}')
    if filename:
        plt.savefig(filename, dpi=300, transparent=True)
    print("Saved", filename)
    

blue = (0, 0.576, 0.902) # 0, 147, 230
green = (0.349,0.745,0.306) # 89, 190, 78
red = (0.984, 0.262, 0.219) # 251, 67, 56 
orange = (0.984, 0.713, 0.305) # 251, 182, 78 
purple = (0.839, 0.286, 0.604) # 214, 73, 1541
anvil = (0.298, 0.78, 0.77) # 76, 199, 196
dark_purple = (0.557, 0, 0.998) # 142, 0, 252
pink = (0.95, 0.78, 0.996) # 242, 199, 154
gray = (0.463,0.463,0.463) # 118, 118, 118

color_profile = [blue, purple, orange, anvil, red]

# csv_files_fus = glob.glob('/project/rcc/hyadav/TransPolymer_3/TransPolymer/final_plots/*.csv')

# rand_files = [file for file in csv_files_fus if 'rand' in file]
# strat_files = [file for file in csv_files_fus if 'strat' in file]

# df_all_rand = {file: pd.read_csv(file) for file in rand_files}
# new_file_names_rand = [os.path.splitext(file)[0] + '_new_2.png' for file in rand_files]

# df_all_strat = {file: pd.read_csv(file) for file in strat_files}
# new_file_names_strat = [os.path.splitext(file)[0] + '_new_2.png' for file in strat_files]

# for idx, (file, df) in enumerate(df_all_rand.items(), start=1):
#     plot_jointplot(df, color_profile[idx-1], filename=new_file_names_rand[idx-1])

# for idx, (file, df) in enumerate(df_all_strat.items(), start=1):
#     plot_jointplot(df, color_profile[idx-1], filename=new_file_names_strat[idx-1])


df = pd.read_csv("/project/rcc/hyadav/TransPolymer_3/TransPolymer/plots/inference_plot_rmse_0.16761838611608945_r2_0.09286379814147949_mae_0.12858445942401886.csv")
plot_jointplot(df, blue, filename="/project/rcc/hyadav/TransPolymer_3/TransPolymer/final_plots/ood_ood_scaled_strat_model_2025.png")