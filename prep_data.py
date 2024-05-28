import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
import pdb
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler
import joblib
from joblib import dump, load

def ce_data_exact_paper_seq_fusion_scaled(train_num_file, train_smiles_file, train_output_file, test_num_file, test_smiles_file, test_output_file):
    
    def extract_info(num_file, smiles_file):
        df_num = pd.read_csv(num_file)
        df_smiles = pd.read_csv(smiles_file)
        df_smiles.fillna('NAN_SMILES', inplace=True)
        df_input = pd.concat([df_num, df_smiles], axis = 1)
        return df_input

    def final_save(df_input, output_file):
        df_input = df_input.applymap(str)   
        df_input['solv_1'] = df_input[['solvent_1_smiles', 'solv_ratio_1']].apply("$".join, axis=1)
        df_input['solv_2'] = df_input[['solvent_2_smiles', 'solv_ratio_2']].apply("$".join, axis=1)
        df_input['solv_3'] = df_input[['solvent_3_smiles', 'solv_ratio_3']].apply("$".join, axis=1)
        df_input['salt_1'] = df_input[['salt_1_smiles', 'salt_1_conc']].apply("$".join, axis=1)
        df_input['salt_2'] = df_input[['salt_2_smiles', 'salt_2_conc']].apply("$".join, axis=1)
        df_input['properties'] = df_input[['protocol','current_density']].apply("$".join, axis=1)
        df_input_final = pd.DataFrame()
        df_input_final['input'] = df_input[['solv_1', 'solv_2', 'solv_3', 'salt_1', 'salt_2', 'properties']].apply("|".join, axis=1)
        df_input_final['log(1-CE)'] = df_input['log(1-CE)']
        df_input_final.to_csv(output_file, index = False)

    df_input_train = extract_info(train_num_file, train_smiles_file)
    df_input_test = extract_info(test_num_file, test_smiles_file)

    num_cols = ['solv_ratio_1', 'solv_ratio_2', 'solv_ratio_3', 'salt_1_conc', 'salt_2_conc', 'protocol','current_density']

    scaler = StandardScaler()
    df_input_train[num_cols] = scaler.fit_transform(df_input_train[num_cols])
    df_input_test[num_cols] = scaler.transform(df_input_test[num_cols])
    dump(scaler, 'std_scaler_ce_train.bin', compress=True)

    sc = StandardScaler()
    df_input_train['log(1-CE)'] = sc.fit_transform(df_input_train['log(1-CE)'].values.reshape(-1, 1))
    df_input_test['log(1-CE)'] = sc.transform(df_input_test['log(1-CE)'].values.reshape(-1, 1))
    dump(sc, 'std_scaler_ce_train_CE.bin', compress=True)
    
    final_save(df_input_train, train_output_file)
    final_save(df_input_test, test_output_file)

def ood_ce_data_exact_paper_seq_fusion_scaled(num_file, smiles_file, output_file):
    df_num = pd.read_csv(num_file)
    df_smiles = pd.read_csv(smiles_file)
    df_smiles.fillna('NAN_SMILES', inplace=True)
    df_input = pd.concat([df_num, df_smiles], axis = 1)

    num_cols = ['solv_ratio_1', 'solv_ratio_2', 'solv_ratio_3', 'salt_1_conc', 'salt_2_conc', 'protocol','current_density']

    # scaler = StandardScaler()
    # df_input[num_cols] = scaler.fit_transform(df_input[num_cols])
    # dump(scaler, 'std_scaler_cood_ood_ce_train.bin', compress=True)

    sc = StandardScaler()
    df_input['log(1-CE)'] = sc.fit_transform(df_input['log(1-CE)'].values.reshape(-1, 1))
    dump(sc, 'std_scaler_cood_ood_ce_train_CE.bin', compress=True)

    df_input['solvent_1_smiles'] = df_input['solv_smile_comb']
    df_input['solvent_2_smiles'] = "NAN_SMILES"
    df_input['solvent_3_smiles'] = "NAN_SMILES"
    df_input['salt_1_smiles'] = df_input['salt_smile_comb']
    df_input['salt_2_smiles'] = "NAN_SMILES"
    
    df_input = df_input.applymap(str)   
    df_input['solv_1'] = df_input[['solvent_1_smiles', 'solv_ratio_1']].apply("$".join, axis=1)
    df_input['solv_2'] = df_input[['solvent_2_smiles', 'solv_ratio_2']].apply("$".join, axis=1)
    df_input['solv_3'] = df_input[['solvent_3_smiles', 'solv_ratio_3']].apply("$".join, axis=1)
    df_input['salt_1'] = df_input[['salt_1_smiles', 'salt_1_conc']].apply("$".join, axis=1)
    df_input['salt_2'] = df_input[['salt_2_smiles', 'salt_2_conc']].apply("$".join, axis=1)
    df_input['properties'] = df_input[['protocol','current_density']].apply("$".join, axis=1)
    df_input_final = pd.DataFrame()
    df_input_final['input'] = df_input[['solv_1', 'solv_2', 'solv_3', 'salt_1', 'salt_2', 'properties']].apply("|".join, axis=1)
    df_input_final['log(1-CE)'] = df_input['log(1-CE)']
    df_input_final.to_csv(output_file, index = False)

def exact_paper_seq_fusion_scaled(train_num_file, train_smiles_file, train_output_file, test_num_file, test_smiles_file, test_output_file):
    df_num_train = pd.read_csv(train_num_file)
    df_smiles_train = pd.read_csv(train_smiles_file)
    df_smiles_train.fillna('NAN_SMILES', inplace=True)
    df_input_train = pd.concat([df_num_train, df_smiles_train], axis = 1)
    df_input_train['conductivity_log'] = np.log10(np.exp(df_input_train['conductivity_log']))

    df_num_test = pd.read_csv(test_num_file)
    df_smiles_test = pd.read_csv(test_smiles_file)
    df_smiles_test.fillna('NAN_SMILES', inplace=True)
    df_input_test = pd.concat([df_num_test, df_smiles_test], axis = 1)
    df_input_test['conductivity_log'] = np.log10(np.exp(df_input_test['conductivity_log']))

    num_cols = ['solv_ratio_1', 'solv_ratio_2', 'solv_ratio_3', 'solv_ratio_4', 'conc_salt', 'temperature']

    ## Random Code
    scaler = StandardScaler()
    df_input_train[num_cols] = scaler.fit_transform(df_input_train[num_cols])
    df_input_test[num_cols] = scaler.transform(df_input_test[num_cols])
    df_input_train[num_cols] = df_input_train[num_cols].round(2)
    df_input_test[num_cols] = df_input_test[num_cols].round(2)
    dump(scaler, 'std_scaler_random_decimal_2.bin', compress=True)

    def final_save(df_input, output_file):
        df_input = df_input.applymap(str)   
        df_input['solv_1'] = df_input[['solv_1_sm', 'solv_ratio_1']].apply("$".join, axis=1)
        df_input['solv_2'] = df_input[['solv_2_sm', 'solv_ratio_2']].apply("$".join, axis=1)
        df_input['solv_3'] = df_input[['solv_3_sm', 'solv_ratio_3']].apply("$".join, axis=1)
        df_input['solv_4'] = df_input[['solv_4_sm', 'solv_ratio_4']].apply("$".join, axis=1)
        df_input['salt'] = df_input[['salt_sm', 'conc_salt']].apply("$".join, axis=1)
        df_input_final = pd.DataFrame()
        df_input_final['input'] = df_input[['solv_1', 'solv_2', 'solv_3', 'solv_4', 'salt']].apply("|".join, axis=1)
        df_input_final['temperature'] = df_input['temperature']
        df_input_final['conductivity_log'] = df_input['conductivity_log']
        df_input_final.to_csv(output_file, index = False)
    
    final_save(df_input_train, train_output_file)
    final_save(df_input_test, test_output_file)

def exact_paper_seq_fusion_scaled_other(train_num_file, train_smiles_file, train_output_file, test_num_file, test_smiles_file, test_output_file):
    df_num_train = pd.read_csv(train_num_file)
    df_smiles_train = pd.read_csv(train_smiles_file)
    df_smiles_train.fillna('NAN_SMILES', inplace=True)
    df_input_train = pd.concat([df_num_train, df_smiles_train], axis = 1)
    df_input_train['conductivity_log'] = np.log10(np.exp(df_input_train['conductivity_log']))

    df_num_test = pd.read_csv(test_num_file)
    df_smiles_test = pd.read_csv(test_smiles_file)
    df_smiles_test.fillna('NAN_SMILES', inplace=True)
    df_input_test = pd.concat([df_num_test, df_smiles_test], axis = 1)
    df_input_test['conductivity_log'] = np.log10(np.exp(df_input_test['conductivity_log']))

    num_cols = ['solv_ratio_1', 'solv_ratio_2', 'solv_ratio_3', 'solv_ratio_4', 'conc_salt', 'temperature']

    scaler = load('std_scaler_random_decimal_2.bin')
    df_input_train[num_cols] = scaler.transform(df_input_train[num_cols])
    df_input_test[num_cols] = scaler.transform(df_input_test[num_cols])

    df_input_train[num_cols] = df_input_train[num_cols].round(2)
    df_input_test[num_cols] = df_input_test[num_cols].round(2)

    sc = load('std_scaler_random_conductivity.bin')
    df_input_train['conductivity_log'] = sc.transform(df_input_train['conductivity_log'].values.reshape(-1, 1))
    df_input_test['conductivity_log'] = sc.transform(df_input_test['conductivity_log'].values.reshape(-1, 1))

    def final_save(df_input, output_file):
        df_input = df_input.applymap(str)   
        df_input['solv_1'] = df_input[['solv_1_sm', 'solv_ratio_1']].apply("$".join, axis=1)
        df_input['solv_2'] = df_input[['solv_2_sm', 'solv_ratio_2']].apply("$".join, axis=1)
        df_input['solv_3'] = df_input[['solv_3_sm', 'solv_ratio_3']].apply("$".join, axis=1)
        df_input['solv_4'] = df_input[['solv_4_sm', 'solv_ratio_4']].apply("$".join, axis=1)
        df_input['salt'] = df_input[['salt_sm', 'conc_salt']].apply("$".join, axis=1)
        df_input_final = pd.DataFrame()
        df_input_final['input'] = df_input[['solv_1', 'solv_2', 'solv_3', 'solv_4', 'salt']].apply("|".join, axis=1)
        df_input_final['temperature'] = df_input['temperature']
        df_input_final['conductivity_log'] = df_input['conductivity_log']
        df_input_final.to_csv(output_file, index = False)
    
    final_save(df_input_train, train_output_file)
    final_save(df_input_test, test_output_file)

def exact_paper_seq_temp_separate(num_file, smiles_file, output_file):
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
    df_input_final['input'] = df_input[['solv_1', 'solv_2', 'solv_3', 'solv_4', 'salt']].apply("|".join, axis=1)  
    df_input_final['temperature'] = df_input['temperature']
    df_input_final['conductivity_log'] = df_input['conductivity_log']

    df_input_final.to_csv(output_file, index = False)

def ood_scaled_fusion(num_file, smiles_file, output_file):
    df_num = pd.read_csv(num_file)
    df_num = df_num.round(2)
    df_smiles = pd.read_csv(smiles_file)
    df_smiles.fillna('NAN_SMILES', inplace=True)
    df_input = pd.concat([df_num, df_smiles], axis = 1)
    df_input['conductivity_log'] = np.log10(np.exp(df_input['conductivity_log']))

    num_cols = ['solv_ratio_1', 'solv_ratio_2', 'solv_ratio_3', 'solv_ratio_4', 'conc_salt', 'temperature']

    # scaler = StandardScaler()
    # df_input[num_cols] = scaler.fit_transform(df_input[num_cols])
    # dump(scaler, 'std_scaler_cond_ood.bin', compress=True)

    sc = StandardScaler()
    df_input['conductivity_log'] = sc.fit_transform(df_input['conductivity_log'].values.reshape(-1, 1))
    dump(sc, 'std_scaler_cond_ood_conductivity_log_decimal_2.bin', compress=True)

    df_input['solv_1_sm'] = df_input['solv_comb_sm']
    df_input['solv_2_sm'] = "NAN_SMILES"
    df_input['solv_3_sm'] = "NAN_SMILES"
    df_input['solv_4_sm'] = "NAN_SMILES"

    df_input = df_input.applymap(str)   
    df_input['solv_1'] = df_input[['solv_1_sm', 'solv_ratio_1']].apply("$".join, axis=1)
    df_input['solv_2'] = df_input[['solv_2_sm', 'solv_ratio_2']].apply("$".join, axis=1)
    df_input['solv_3'] = df_input[['solv_3_sm', 'solv_ratio_3']].apply("$".join, axis=1)
    df_input['solv_4'] = df_input[['solv_4_sm', 'solv_ratio_4']].apply("$".join, axis=1)
    df_input['salt'] = df_input[['salt_sm', 'conc_salt']].apply("$".join, axis=1)
    
    df_input_final = pd.DataFrame()
    df_input_final['input'] = df_input[['solv_1', 'solv_2', 'solv_3', 'solv_4', 'salt']].apply("|".join, axis=1)
    df_input_final['temperature'] = df_input['temperature']
    df_input_final['conductivity_log'] = df_input['conductivity_log']

    df_input_final.to_csv(output_file, index = False)
    

def main():
    ood_ce_data_exact_paper_seq_fusion_scaled("./data/CE-data/ce_ood_final_add.csv", "./data/CE-data/ce_ood_final_comb.csv", "./data/CE-data/ce_ood_exact_paper_seq_unscaled_nonfusion_no_mol_new.csv")
    # ood_scaled_fusion("./data/cond_ood_final_add.csv", "./data/cond_ood_final_comp.csv", "./data/cond_ood_exact_paper_seq_unscaled_fusion_no_mol_new_decimal_2.csv")
    # exact_paper_seq_temp_separate("./data/freqI_train_multi_comp_add.csv", "./data/freqI_train_multi_comp.csv", "./data/freqI_train_exact_paper_seq_common_log_temp_sep.csv")
    # exact_paper_seq("./data/random_train_multi_add.csv", "./data/random_train_multi_comp.csv", "./data/random_train_exact_paper_seq.csv")
    # exact_paper_seq("./data/random_test_multi_add.csv", "./data/random_test_multi_comp.csv", "./data/random_test_exact_paper_seq_common_log_temp_concat.csv")
    # exact_paper_seq_ood_using_all_solvs_drop_negative("./data/ood_add.csv", "./data/ood_comp.csv", "./data/ood_exact_paper_seq_all_solvs_common_log_no_negative_conductivity_temp_sep.csv")
    # exact_paper_seq_ood_using_all_solvs("./data/ood_add.csv", "./data/ood_comp.csv", "./data/ood_exact_paper_seq_ignoring_temperature.csv")
    # exact_paper_seq_ood_using_all_solvs_temp("./data/ood_add.csv", "./data/ood_comp.csv", "./data/ood_exact_paper_seq_all_solvs_common_log_ignoring_temperature_temp_sep.csv")
    # exact_paper_seq_fusion_scaled_other("./data/freqII_train_multi_comp_add.csv", "./data/freqII_train_multi_comp.csv", "./data/freqII_train_exact_paper_seq_common_log_fusion_scaled_no_mol_new_decimal_2.csv", 
    #                               "./data/freqII_test_multi_comp_add.csv", "./data/freqII_test_multi_comp.csv", "./data/freqII_test_exact_paper_seq_common_log_fusion_scaled_no_mol_new_decimal_2.csv")
    # exact_paper_seq_fusion_scaled("./data/random_train_multi_add.csv", "./data/random_train_multi_comp.csv", "./data/random_train_exact_all_scaled_fusion_no_mol_new_decimal_2.csv", 
    #                               "./data/random_test_multi_add.csv", "./data/random_test_multi_comp.csv", "./data/random_test_exact_all_scaled_fusion_no_mol_new_decimal_2.csv")
    # ce_data_exact_paper_seq_fusion_scaled("./data/CE-data/train_CE_add.csv", "./data/CE-data/train_CE_comb_comm.csv", "./data/CE-data/train_CE_exact_all_scaled_nonfusion_no_mol_new.csv", 
    #                               "./data/CE-data/test_CE_add.csv", "./data/CE-data/test_CE_comb_comm.csv", "./data/CE-data/test_CE_exact_all_scaled_nonfusion_no_mol_new.csv")
    

if __name__ == "__main__":
    main()