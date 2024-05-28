import pandas as pd
import pdb 
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

# random_train = pd.read_csv('data/random_train_exact_paper_seq_common_log.csv')
# random_test = pd.read_csv( 'data/random_test_exact_paper_seq_common_log.csv')
# freqI_train = pd.read_csv('data/freqI_train_exact_paper_seq_common_log.csv' )
# freqI_test = pd.read_csv('data/freqI_test_exact_paper_seq_common_log.csv')
# freqII_train = pd.read_csv('data/freqII_train_exact_paper_seq_common_log.csv')
# freqII_test = pd.read_csv('data/freqII_test_exact_paper_seq_common_log.csv')
# OOD = pd.read_csv('data/ood_exact_paper_seq_all_solvs_common_log.csv')

def count_zeros(row):
    return row.eq(0).sum()

def solvent_counts(df):
    df['zeros_count'] = df[['solv_ratio_1', 'solv_ratio_2', 'solv_ratio_3', 'solv_ratio_4']].apply(count_zeros, axis=1)
    counts = df.groupby('zeros_count').size()
    return df, counts

def solvent_counts_main(datasets):
    counts_arr = []

    for i in range(len(datasets)):
        df, counts = solvent_counts(datasets[i])
        filename = f"./data/study_data/dataset_{i}.csv"
        df.to_csv(filename, index=False)
        counts_arr.append(counts)

    counts_df = pd.DataFrame(counts_arr)
    counts_df.fillna('0', inplace=True)
    counts_df['Dataset'] = ['random_train', 'random_test', 'freqI_train', 'freqI_test', 'freqII_train', 'freqII_test', 'OOD_data']
    counts_df = pd.concat([counts_df.iloc[:, -1:], counts_df.iloc[:, :-1]], axis=1)
    print("Counts of rows w.r.t 0 solvent ratios")
    print(counts_df)

def unique(datasets, datasets_sm):
    lengths = []
    cat_1 = []
    cat_2 = []
    cat_3 = []
    
    for i in range(len(datasets)):
        dataset = datasets[i]
        dataset_sm = datasets_sm[i]
        df = pd.DataFrame()
        if i < 6: 
            df['solv_1_sm'] = dataset_sm['solv_1_sm']
            df['solv_2_sm'] = dataset_sm['solv_2_sm']
            df['solv_3_sm'] = dataset_sm['solv_3_sm']
            df['solv_4_sm'] = dataset_sm['solv_4_sm']
            df['salt_sm'] = dataset_sm['salt_sm']
            df['conc_salt'] = dataset['conc_salt']
            df['temperature'] = dataset['temperature']

            # print("Total Number of Rows: ", len(df), "in dataset: ", i)
            lengths.append(len(datasets[i]))

            #Category 1: Rows with unique solvents
            unique_rows = df.drop_duplicates(subset=['solv_1_sm', 'solv_2_sm', 'solv_3_sm', 'solv_4_sm'])
            cat_1.append(len(unique_rows))

            #Category 2: Rows with unique combinations of salts and solvents
            unique_rows = df.drop_duplicates(subset=['solv_1_sm', 'solv_2_sm', 'solv_3_sm', 'solv_4_sm', 'salt_sm'])
            cat_2.append(len(unique_rows))

            #Category 3: Rows with unique combinations of salts, solvents and salt concentraion
            unique_rows = df.drop_duplicates(subset=['solv_1_sm', 'solv_2_sm', 'solv_3_sm', 'solv_4_sm', 'salt_sm', 'conc_salt'])
            cat_3.append(len(unique_rows))


        elif i == 6:
            df['solv_comb_sm'] = dataset_sm['solv_comb_sm']
            df['salt_sm'] = dataset_sm['salt_sm']
            df['conc_salt'] = dataset['conc_salt']
            df['temperature'] = dataset['temperature']

            # print("Total Number of Rows: ", len(df), "in dataset: ", i)
            lengths.append(len(datasets[i]))

            #Category 1: Rows with unique solvents
            unique_rows = df.drop_duplicates(subset=['solv_comb_sm'])
            cat_1.append(len(unique_rows))

            #Category 2: Rows with unique combinations of salts and solvents
            unique_rows = df.drop_duplicates(subset=['solv_comb_sm', 'salt_sm'])
            cat_2.append(len(unique_rows))

            #Category 3: Rows with unique combinations of salts, solvents and salt concentraion
            unique_rows = df.drop_duplicates(subset=['solv_comb_sm', 'salt_sm', 'conc_salt'])
            cat_3.append(len(unique_rows))

    categories_df = pd.DataFrame()
    categories_df['Dataset'] = ['random_train', 'random_test', 'freqI_train', 'freqI_test', 'freqII_train', 'freqII_test', 'OOD']
    categories_df['Lengths'] = lengths
    categories_df['Category 1'] = [x / y for x, y in zip(cat_1, lengths)]
    categories_df['Category 2'] = [x / y for x, y in zip(cat_2, lengths)]   
    categories_df['Category 3'] = [x / y for x, y in zip(cat_3, lengths)]   

    print(categories_df)
    categories_df.to_csv("./data/study_data/categories.csv", index=False)

def study_ood_predictions(df_num, df_sm, df):

    # pdb.set_trace()

    # Combine values of Column1 and Column2 into a new column
    df_sm['Combined'] = df_sm['solv_comb_sm'] + df_sm['salt_sm'].astype(str)

    # Group by the combined column and assign group number to each row
    df_sm['Group'] = df_sm.groupby('Combined').ngroup() + 1

    # Drop the combined column if not needed
    df_sm.drop('Combined', axis=1, inplace=True)

    x = np.arange(0, df.shape[0])
    y_temp = df_num['temperature']
    y_true = df['conductivity_log']
    y_pred = df['predictions']
    y_error = df['rmse_per_point']
    y_combo = df_sm['Group']

    # plt.figure()
    # plt.plot(x, y_true, color = 'green', label = 'True')
    # plt.plot(x, y_pred, color = 'blue', label = 'Pred')
    # # plt.plot(x, y_error, color = 'red', label = 'RMSE Per Point')
    # # plt.plot(x, y_temp, color = 'purple', markersize = '1')
    # plt.legend()
    # filename = ("./data/study_data/OOD_predictions_scaled.png")
    # plt.savefig(filename)

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    # ax1.set_ylabel('Ionic Conductivity', color=color)
    # ax1.plot(x, y_true, color = 'yellow', label = 'True')
    # ax1.plot(x, y_pred, color = 'orange', label = 'Pred')
    # ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylabel('Temperature', color=color)
    ax1.plot(x, y_temp, color = color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Combo Group', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, y_combo, color = color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend()
    filename = ("./data/study_data/OOD_temp_vs_group.png")
    plt.savefig(filename)


# random_train = pd.read_csv("./data/random_train_multi_add.csv")
# random_test = pd.read_csv("./data/random_test_multi_add.csv")
# freqI_train = pd.read_csv("./data/freqI_train_multi_comp_add.csv")
# freqI_test = pd.read_csv("./data/freqI_test_multi_comp_add.csv")
# freqII_train = pd.read_csv("./data/freqII_train_multi_comp_add.csv")
# freqII_test = pd.read_csv("./data/freqII_test_multi_comp_add.csv")
OOD_test = pd.read_csv("./data/ood_add.csv")

# random_train_sm = pd.read_csv("./data/random_train_multi_comp.csv")
# random_test_sm = pd.read_csv("./data/random_test_multi_comp.csv")
# freqI_train_sm = pd.read_csv("./data/freqI_train_multi_comp.csv")
# freqI_test_sm = pd.read_csv("./data/freqI_test_multi_comp.csv")
# freqII_train_sm = pd.read_csv("./data/freqII_train_multi_comp.csv")
# freqII_test_sm = pd.read_csv("./data/freqII_test_multi_comp.csv")
OOD_sm = pd.read_csv("./data/ood_comp.csv")

# datasets = [random_train, random_test, freqI_train, freqI_test, freqII_train, freqII_test, OOD_test]
# datasets_sm = [random_train_sm, random_test_sm, freqI_train_sm, freqI_test_sm, freqII_train_sm, freqII_test_sm, OOD_sm]

# unique(datasets, datasets_sm)

df = pd.read_csv("./data/study_data/ood_exact_paper_seq_all_solvs_common_log_scaled_results.csv")
study_ood_predictions(OOD_test, OOD_sm, df)
