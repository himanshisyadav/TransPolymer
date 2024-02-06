import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
import pdb
from sklearn.preprocessing import QuantileTransformer

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

    # pdb.set_trace()
    #Using only unique solvents and salts, temperature is ignored
    unique_rows = df_input.drop_duplicates(subset=['solv_ratio_1', 'solv_ratio_2', 'solv_ratio_3', 'solv_ratio_4','mol_wt_solv_1', 'mol_wt_solv_2', 'mol_wt_solv_3', 'mol_wt_solv_4','mol_wt_salt', 'conc_salt', 'solv_comb_sm', 'salt_sm'])
    df_input = unique_rows

    # #Using only unique solvents and salts
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
    df_input = df_input[df_input['conductivity_log'] <= 0]

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

def main():
    # all_dollar()
    # exact_paper_seq("./data/random_train_multi_add.csv", "./data/random_train_multi_comp.csv", "./data/random_train_exact_paper_seq_common_log.csv")
    # exact_paper_seq("./data/random_test_multi_add.csv", "./data/random_test_multi_comp.csv", "./data/random_test_exact_paper_seq_common_log.csv")
    # exact_paper_seq_ood_using_all_solvs_drop_negative("./data/ood_add.csv", "./data/ood_comp.csv", "./data/ood_exact_paper_seq_all_solvs_no_positive_conductivity.csv")
    exact_paper_seq_ood_using_all_solvs("./data/ood_add.csv", "./data/ood_comp.csv", "./data/ood_exact_paper_seq_ignoring_temperature.csv")
    

if __name__ == "__main__":
    main()