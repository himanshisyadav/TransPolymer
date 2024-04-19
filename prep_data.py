import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
import pdb
from sklearn.preprocessing import QuantileTransformer
from itertools import permutations
from sklearn.preprocessing import StandardScaler

def only_smiles(smiles_file, output_file):
    df_smiles = pd.read_csv(smiles_file)
    df_smiles.fillna('NAN_SMILES', inplace=True)
    df_input_final = pd.DataFrame()
    df_input_final['input'] = df_smiles[['solv_comb_sm', 'salt_sm']].apply("|".join, axis=1)
    df_input_final['conductivity_log'] = df_smiles['conductivity_log']
    df_input_final.to_csv(output_file, index = False)

def smiles_and_temp(num_file, smiles_file, output_file):
    df_smiles = pd.read_csv(smiles_file)
    df_num = pd.read_csv(num_file)
    df_smiles.fillna('NAN_SMILES', inplace=True)
    df_input = pd.concat([df_num, df_smiles], axis = 1)
    df_input = df_input.applymap(str)
    df_input_final = pd.DataFrame()
    df_input_final['input'] = df_input[['solv_comb_sm', 'salt_sm', 'temperature']].apply("|".join, axis=1)
    df_input_final['conductivity_log'] = df_smiles['conductivity_log']
    df_input_final.to_csv(output_file, index = False)

def all_dollar():
    df_num = pd.read_csv("./data/ood_multi_comp_add.csv")
    df_num = df_num.round(2)
    df_smiles = pd.read_csv("./data/ood_multi_comp_comb.csv")
    df_smiles.fillna('NAN_SMILES', inplace=True)
    df_input = pd.concat([df_num, df_smiles], axis = 1)
    df_input = df_input.applymap(str)
    df_input_final = pd.DataFrame()
    df_input_final['input'] = df_input[['solv_comb_sm', 'salt_sm','solv_ratio_1', 'solv_ratio_2', 'solv_ratio_3', 'solv_ratio_4', 'mol_wt_solv_1', 'mol_wt_solv_2', 'mol_wt_solv_3', 'mol_wt_solv_4', 'conc_salt', 'temperature']].apply("$".join, axis=1)
    df_input_final['conductivity_log'] = df_input['conductivity_log']

    train, test = train_test_split(df_input_final, test_size=0.2)

    print(train.shape[0], test.shape[0])

    df_input_final.to_csv("./data/ood_decimal_2_for_inference.csv", index = False)
    # train.to_csv("./data/our_train_combined.csv", index = False)
    # test.to_csv("./data/our_test_combined.csv", index = False)

def all_pipe(num_file, smiles_file, output_file):
    df_num = pd.read_csv(num_file)
    df_smiles = pd.read_csv(smiles_file)
    df_smiles.fillna('NAN_SMILES', inplace=True)
    df_input = pd.concat([df_num, df_smiles], axis = 1)
    df_input = df_input.applymap(str)
    df_input_final = pd.DataFrame()
    df_input_final['input'] = df_input[['solv_ratio_1', 'solv_ratio_2', 'solv_ratio_3', 'solv_ratio_4', 'mol_wt_solv_1', 'mol_wt_solv_2', 'mol_wt_solv_3', 'mol_wt_solv_4', 'conc_salt', 'temperature', 'solv_comb_sm', 'salt_sm']].apply("|".join, axis=1)
    df_input_final['conductivity_log'] = df_input['conductivity_log']
    df_input_final.to_csv(output_file, index = False)

def exact_paper_seq(num_file, smiles_file, output_file):
    df_num = pd.read_csv(num_file)
    df_smiles = pd.read_csv(smiles_file)
    df_smiles.fillna('NAN_SMILES', inplace=True)
    df_input = pd.concat([df_num, df_smiles], axis = 1)
    df_input['conductivity_log'] = np.log10(np.exp(df_input['conductivity_log']))

    # #Using Quantile Transformer for Normalisation
    # num_cols = [ 'solv_ratio_1', 'solv_ratio_2', 'solv_ratio_3', 'solv_ratio_4', 'mol_wt_solv_1', 'mol_wt_solv_2', 'mol_wt_solv_3', 'mol_wt_solv_4', 'conc_salt','temperature' ]
    # for col_name in num_cols:
    #     num_df = df_input[col_name]
    #     num_df.replace('', 0, inplace=True)
    #     col_values = num_df.values.reshape(-1, 1)
    #     numerical_transformer = QuantileTransformer(output_distribution='normal')
    #     col_values_norm = numerical_transformer.fit_transform(col_values)
    #     df_input[col_name] = col_values_norm
    #     df_input[col_name] = df_input[col_name].round(2)

    # df_num_new = df_input[num_cols].copy()
    # df_num_new.to_csv(num_vocab_file)

    df_input = df_input.applymap(str)   
    df_input['solv_1'] = df_input[['solv_1_sm', 'solv_ratio_1', 'mol_wt_solv_1']].apply("$".join, axis=1)
    df_input['solv_2'] = df_input[['solv_2_sm', 'solv_ratio_2', 'mol_wt_solv_2']].apply("$".join, axis=1)
    df_input['solv_3'] = df_input[['solv_3_sm', 'solv_ratio_3', 'mol_wt_solv_3']].apply("$".join, axis=1)
    df_input['solv_4'] = df_input[['solv_4_sm', 'solv_ratio_4', 'mol_wt_solv_4']].apply("$".join, axis=1)
    df_input['salt'] = df_input[['salt_sm', 'conc_salt']].apply("$".join, axis=1)
    df_input_final = pd.DataFrame()
    df_input_final['input'] = df_input[['solv_1', 'solv_2', 'solv_3', 'solv_4', 'salt', 'temperature']].apply("|".join, axis=1)
    df_input_final['conductivity_log'] = df_input['conductivity_log']
    df_input_final.to_csv(output_file, index = False)

def exact_paper_seq_decimal_2_common_log(num_file, smiles_file, output_file):
    df_num = pd.read_csv(num_file)
    df_smiles = pd.read_csv(smiles_file)
    df_smiles.fillna('NAN_SMILES', inplace=True)
    df_input = pd.concat([df_num, df_smiles], axis = 1)
    df_input['conductivity_log'] = np.log10(np.exp(df_input['conductivity_log']))

    num_cols = [ 'solv_ratio_1', 'solv_ratio_2', 'solv_ratio_3', 'solv_ratio_4', 'mol_wt_solv_1', 'mol_wt_solv_2', 'mol_wt_solv_3', 'mol_wt_solv_4', 'conc_salt','temperature','conductivity_log']
    for col_name in num_cols:
        df_input[col_name] = df_input[col_name].round(2)

    df_input = df_input.applymap(str)   
    df_input['solv_1'] = df_input[['solv_1_sm', 'solv_ratio_1', 'mol_wt_solv_1']].apply("$".join, axis=1)
    df_input['solv_2'] = df_input[['solv_2_sm', 'solv_ratio_2', 'mol_wt_solv_2']].apply("$".join, axis=1)
    df_input['solv_3'] = df_input[['solv_3_sm', 'solv_ratio_3', 'mol_wt_solv_3']].apply("$".join, axis=1)
    df_input['solv_4'] = df_input[['solv_4_sm', 'solv_ratio_4', 'mol_wt_solv_4']].apply("$".join, axis=1)
    df_input['salt'] = df_input[['salt_sm', 'conc_salt']].apply("$".join, axis=1)
    df_input_final = pd.DataFrame()
    df_input_final['input'] = df_input[['solv_1', 'solv_2', 'solv_3', 'solv_4', 'salt', 'temperature']].apply("|".join, axis=1)
    df_input_final['conductivity_log'] = df_input['conductivity_log']
    df_input_final.to_csv(output_file, index = False)

def exact_paper_seq_common_log_data_aug(num_file, smiles_file, output_file):
    df_num = pd.read_csv(num_file)
    df_smiles = pd.read_csv(smiles_file)
    df_smiles.fillna('NAN_SMILES', inplace=True)
    df_input = pd.concat([df_num, df_smiles], axis = 1)
    df_input['conductivity_log'] = np.log10(np.exp(df_input['conductivity_log']))

    df_input = df_input.applymap(str)   
    df_input['solv_1'] = df_input[['solv_1_sm', 'solv_ratio_1', 'mol_wt_solv_1']].apply("$".join, axis=1)
    df_input['solv_2'] = df_input[['solv_2_sm', 'solv_ratio_2', 'mol_wt_solv_2']].apply("$".join, axis=1)
    df_input['solv_3'] = df_input[['solv_3_sm', 'solv_ratio_3', 'mol_wt_solv_3']].apply("$".join, axis=1)
    df_input['solv_4'] = df_input[['solv_4_sm', 'solv_ratio_4', 'mol_wt_solv_4']].apply("$".join, axis=1)
    df_input['salt'] = df_input[['salt_sm', 'conc_salt']].apply("$".join, axis=1)

    def unique_permutations(row):
        return list(permutations(row[:4]))
    
    df_input_final = pd.DataFrame()
    df_input_final = df_input[['solv_1', 'solv_2', 'solv_3', 'solv_4', 'salt', 'temperature', 'conductivity_log']]

    permutations_lists = df_input_final.apply(unique_permutations, axis=1)

    permutations_data = []
    for i, row in enumerate(permutations_lists):
        for perm in row:
            permutations_data.append(list(perm) + [df_input_final.iloc[i]['salt']] + [df_input_final.iloc[i]['temperature']] + [df_input_final.iloc[i]['conductivity_log']])

    permutations_df = pd.DataFrame(permutations_data, columns=['solv_1', 'solv_2', 'solv_3', 'solv_4', 'salt', 'temperature', 'conductivity_log'])
    permutations_df = permutations_df.drop_duplicates()

    final_df = pd.DataFrame()
    final_df['input'] = permutations_df[['solv_1', 'solv_2', 'solv_3', 'solv_4', 'salt', 'temperature']].apply("|".join, axis=1)
    final_df['conductivity_log'] = permutations_df['conductivity_log']

    pdb.set_trace()

def exact_paper_seq_common_log_temp_first_token(num_file, smiles_file, output_file):
    df_num = pd.read_csv(num_file)
    df_smiles = pd.read_csv(smiles_file)
    df_smiles.fillna('NAN_SMILES', inplace=True)
    df_input = pd.concat([df_num, df_smiles], axis = 1)
    df_input['conductivity_log'] = np.log10(np.exp(df_input['conductivity_log']))

    df_input = df_input.applymap(str)   
    df_input['solv_1'] = df_input[['solv_1_sm', 'solv_ratio_1', 'mol_wt_solv_1']].apply("$".join, axis=1)
    df_input['solv_2'] = df_input[['solv_2_sm', 'solv_ratio_2', 'mol_wt_solv_2']].apply("$".join, axis=1)
    df_input['solv_3'] = df_input[['solv_3_sm', 'solv_ratio_3', 'mol_wt_solv_3']].apply("$".join, axis=1)
    df_input['solv_4'] = df_input[['solv_4_sm', 'solv_ratio_4', 'mol_wt_solv_4']].apply("$".join, axis=1)
    df_input['salt'] = df_input[['salt_sm', 'conc_salt']].apply("$".join, axis=1)
    df_input_final = pd.DataFrame()
    df_input_final['input'] = df_input[['temperature','solv_1', 'solv_2', 'solv_3', 'solv_4', 'salt']].apply("|".join, axis=1)
    df_input_final['conductivity_log'] = df_input['conductivity_log']
    df_input_final.to_csv(output_file, index = False)

def exact_paper_seq_common_log_temp_first_token_salt_next(num_file, smiles_file, output_file):
    df_num = pd.read_csv(num_file)
    df_smiles = pd.read_csv(smiles_file)
    df_smiles.fillna('NAN_SMILES', inplace=True)
    df_input = pd.concat([df_num, df_smiles], axis = 1)
    df_input['conductivity_log'] = np.log10(np.exp(df_input['conductivity_log']))
    scaler = StandardScaler()
    # df_input['temperature'] = scaler.fit_transform(df_input['temperature'].values.reshape(-1, 1))

    num_cols = [ 'solv_ratio_1', 'solv_ratio_2', 'solv_ratio_3', 'solv_ratio_4', 'mol_wt_solv_1', 'mol_wt_solv_2', 'mol_wt_solv_3', 'mol_wt_solv_4', 'conc_salt','temperature' ]
    for col_name in num_cols:
        df_input[col_name] = scaler.fit_transform(df_input[col_name].values.reshape(-1, 1))

    df_input = df_input.applymap(str)   
    df_input['solv_1'] = df_input[['solv_1_sm', 'solv_ratio_1', 'mol_wt_solv_1']].apply("$".join, axis=1)
    df_input['solv_2'] = df_input[['solv_2_sm', 'solv_ratio_2', 'mol_wt_solv_2']].apply("$".join, axis=1)
    df_input['solv_3'] = df_input[['solv_3_sm', 'solv_ratio_3', 'mol_wt_solv_3']].apply("$".join, axis=1)
    df_input['solv_4'] = df_input[['solv_4_sm', 'solv_ratio_4', 'mol_wt_solv_4']].apply("$".join, axis=1)
    df_input['salt'] = df_input[['salt_sm', 'conc_salt']].apply("$".join, axis=1)
    df_input_final = pd.DataFrame()
    print(df_input.max(axis=0))
    print(df_input.min(axis=0))
    df_input_final['input'] = df_input[['temperature', 'salt' ,'solv_1', 'solv_2', 'solv_3', 'solv_4']].apply("|".join, axis=1)
    df_input_final['conductivity_log'] = df_input['conductivity_log']
    df_input_final.to_csv(output_file, index = False)

def exact_paper_seq_ood(num_file, smiles_file, output_file):
    df_num = pd.read_csv(num_file)
    # df_num = df_num.round(2)
    df_smiles = pd.read_csv(smiles_file)
    df_smiles.fillna('NAN_SMILES', inplace=True)
    df_input = pd.concat([df_num, df_smiles], axis = 1)

    # unique_rows = df_input.drop_duplicates(subset=['solv_comb_sm','salt_sm'])

    # df_input = unique_rows

    # pdb.set_trace()

    #Using Quantile Transformer for Normalisation
    # num_cols = [ 'solv_ratio_1', 'solv_ratio_2', 'solv_ratio_3', 'solv_ratio_4', 'mol_wt_solv_1', 'mol_wt_solv_2', 'mol_wt_solv_3', 'mol_wt_solv_4', 'conc_salt','temperature' ]
    # for col_name in num_cols:
    #     num_df = df_input[col_name]
    #     num_df.replace('', 0, inplace=True)
    #     # col_values = num_df.values.reshape(-1, 1)
    #     # numerical_transformer = QuantileTransformer(output_distribution='normal')
    #     # col_values_norm = numerical_transformer.fit_transform(col_values)
    #     # df_input[col_name] = col_values_norm
    #     df_input[col_name] = df_input[col_name].round(2)

    df_input = df_input.applymap(str)   

    df_input['solv_1'] = df_input[['solv_comb_sm', 'solv_ratio_1', 'mol_wt_solv_1']].apply("$".join, axis=1)
    df_input['salt'] = df_input[['salt_sm', 'conc_salt']].apply("$".join, axis=1)
    df_input_final = pd.DataFrame()
    df_input_final['input'] = df_input[['solv_1', 'salt', 'temperature']].apply("|".join, axis=1)
    df_input_final['conductivity_log'] = df_input['conductivity_log']
    df_input_final.to_csv(output_file, index = False)

def exact_paper_seq_ood_using_all_solvs(num_file, smiles_file, output_file):
    df_num = pd.read_csv(num_file)
    df_smiles = pd.read_csv(smiles_file)
    df_smiles.fillna('NAN_SMILES', inplace=True)
    df_input = pd.concat([df_num, df_smiles], axis = 1)
    df_input['conductivity_log'] = np.log10(np.exp(df_input['conductivity_log']))

    #Decimal 2 code
    # num_cols = [ 'solv_ratio_1', 'solv_ratio_2', 'solv_ratio_3', 'solv_ratio_4', 'mol_wt_solv_1', 'mol_wt_solv_2', 'mol_wt_solv_3', 'mol_wt_solv_4', 'conc_salt','temperature','conductivity_log']
    # for col_name in num_cols:
    #     df_input[col_name] = df_input[col_name].round(2)

    #Using only unique solvents and salts, temperature is ignored
    # unique_rows = df_input.drop_duplicates(subset=['solv_ratio_1', 'solv_ratio_2', 'solv_ratio_3', 'solv_ratio_4','mol_wt_solv_1', 'mol_wt_solv_2', 'mol_wt_solv_3', 'mol_wt_solv_4','mol_wt_salt', 'conc_salt', 'solv_comb_sm', 'salt_sm'])
    # df_input = unique_rows

    #Using only unique solvents and salts
    # unique_rows = df_input.drop_duplicates(subset=['solv_comb_sm','salt_sm'])
    # df_input = unique_rows

    df_input['solv_1_sm'] = df_input['solv_comb_sm']
    df_input['solv_2_sm'] = "NAN_SMILES"
    df_input['solv_3_sm'] = "NAN_SMILES"
    df_input['solv_4_sm'] = "NAN_SMILES"

    df_input = df_input.applymap(str)   
    df_input['solv_1'] = df_input[['solv_1_sm', 'solv_ratio_1', 'mol_wt_solv_1']].apply("$".join, axis=1)
    df_input['solv_2'] = df_input[['solv_2_sm', 'solv_ratio_2', 'mol_wt_solv_2']].apply("$".join, axis=1)
    df_input['solv_3'] = df_input[['solv_3_sm', 'solv_ratio_3', 'mol_wt_solv_3']].apply("$".join, axis=1)
    df_input['solv_4'] = df_input[['solv_4_sm', 'solv_ratio_4', 'mol_wt_solv_4']].apply("$".join, axis=1)
    df_input['salt'] = df_input[['salt_sm', 'conc_salt']].apply("$".join, axis=1)
    df_input_final = pd.DataFrame()
    df_input_final['input'] = df_input[['solv_1', 'solv_2', 'solv_3', 'solv_4', 'salt', 'temperature']].apply("|".join, axis=1)

    df_input_final['conductivity_log'] = df_input['conductivity_log']
    df_input_final.to_csv(output_file, index = False)


def exact_paper_seq_ood_using_all_solvs_drop_negative(num_file, smiles_file, output_file):
    df_num = pd.read_csv(num_file)
    df_smiles = pd.read_csv(smiles_file)
    df_smiles.fillna('NAN_SMILES', inplace=True)
    df_input = pd.concat([df_num, df_smiles], axis = 1)
    df_input['conductivity_log'] = np.log10(np.exp(df_input['conductivity_log']))
    df_input = df_input[df_input['conductivity_log'] >= 0]

    #Decimal 2 code
    # num_cols = [ 'solv_ratio_1', 'solv_ratio_2', 'solv_ratio_3', 'solv_ratio_4', 'mol_wt_solv_1', 'mol_wt_solv_2', 'mol_wt_solv_3', 'mol_wt_solv_4', 'conc_salt','temperature','conductivity_log']
    # for col_name in num_cols:
    #     df_input[col_name] = df_input[col_name].round(2)

    df_input['solv_1_sm'] = df_input['solv_comb_sm']
    df_input['solv_2_sm'] = "NAN_SMILES"
    df_input['solv_3_sm'] = "NAN_SMILES"
    df_input['solv_4_sm'] = "NAN_SMILES"

    df_input = df_input.applymap(str)   
    df_input['solv_1'] = df_input[['solv_1_sm', 'solv_ratio_1', 'mol_wt_solv_1']].apply("$".join, axis=1)
    df_input['solv_2'] = df_input[['solv_2_sm', 'solv_ratio_2', 'mol_wt_solv_2']].apply("$".join, axis=1)
    df_input['solv_3'] = df_input[['solv_3_sm', 'solv_ratio_3', 'mol_wt_solv_3']].apply("$".join, axis=1)
    df_input['solv_4'] = df_input[['solv_4_sm', 'solv_ratio_4', 'mol_wt_solv_4']].apply("$".join, axis=1)
    df_input['salt'] = df_input[['salt_sm', 'conc_salt']].apply("$".join, axis=1)
    df_input_final = pd.DataFrame()
    df_input_final['input'] = df_input[['temperature', 'salt', 'solv_1', 'solv_2', 'solv_3', 'solv_4' ]].apply("|".join, axis=1)
    df_input_final['conductivity_log'] = df_input['conductivity_log']
    df_input_final.to_csv(output_file, index = False)

def main():
    # all_dollar()
    # exact_paper_seq_common_log_data_aug("./data/random_train_multi_add.csv", "./data/random_train_multi_comp.csv", "./data/random_train_augmented.csv")
    # exact_paper_seq_common_log_temp_first_token_salt_next("./data/random_train_multi_add.csv", "./data/random_train_multi_comp.csv","./data/random_train_temp_first_token_salt_next.csv")
    # exact_paper_seq_common_log_temp_first_token_salt_next("./data/freqII_train_multi_comp_add.csv", "./data/freqII_train_multi_comp.csv","./data/freqII_train_temp_first_token_salt_next_all_scaled.csv")
    # exact_paper_seq("./data/freqII_train_multi_comp_add.csv", "./data/freqII_train_multi_comp.csv", "./data/freqII_train_exact_paper_seq_common_log.csv")
    # exact_paper_seq_decimal_2_common_log("./data/freqII_test_multi_comp_add.csv", "./data/freqII_test_multi_comp.csv", "./data/freqII_test_exact_paper_seq_decimal_2_common_log.csv")
    # exact_paper_seq_ood_using_all_solvs_drop_negative("./data/ood_add.csv", "./data/ood_comp.csv", "./data/ood_exact_paper_seq_all_solvs_temp_first_token_salt_next_no_negative_conductivity_common_log.csv")
    # exact_paper_seq_ood_using_all_solvs("./data/ood_add.csv", "./data/ood_comp.csv", "./data/ood_exact_paper_seq_all_solvs_temp_first_token_salt_next_unique_smiles.csv")
    exact_paper_seq_ood_using_all_solvs("./data/electrolyte-data/electrolyte-data/ood_add.csv", "./data/electrolyte-data/electrolyte-data/ood_comp.csv", "./data/ood_exact_paper_seq_all_solvs_common_log_new.csv")
    
if __name__ == "__main__":
    main()  