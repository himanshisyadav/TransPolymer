import pandas as pd
import numpy as np
import pdb

def check_smiles():
    ood_smiles = pd.read_csv("./data/ood_multi_comp_comb.csv")
    random_smiles = pd.read_csv("./data/random_train_multi_comp.csv")

    ood_smiles['combined_sm'] = ood_smiles['solv_comb_sm'] + ood_smiles['salt_sm']
    ood_unique = ood_smiles['combined_sm'].unique()

    random_smiles['combined_sm'] = random_smiles['solv_1_sm'] + random_smiles['solv_2_sm'] + random_smiles['solv_3_sm'] + random_smiles['solv_4_sm'] + random_smiles['salt_sm']
    random_unique = random_smiles['combined_sm'].unique()

    ood_unique_list = ood_unique.tolist()
    random_unique_list = random_unique.tolist()

    diff = np.setdiff1d(ood_unique_list,random_unique_list)

def check_discrete():
    vocab = pd.read_csv('./data/vocab/vocab_sup_PE_I.csv')
    random_train = pd.read_csv("./data/random_train_multi_add.csv")
    random_train = random_train.round(2)

    random_train['combined'] = random_train['solv_ratio_1'] + random_train['solv_ratio_2'] + random_train['solv_ratio_3'] + random_train['solv_ratio_4'] + random_train['mol_wt_solv_1'] + random_train['mol_wt_solv_2'] + random_train['mol_wt_solv_3'] + random_train['mol_wt_solv_4'] + random_train['conc_salt'] + random_train['temperature']
    random_train_unique = random_train['combined'].unique()
    random_train_unique_list = random_train_unique.tolist()
    vocab_unique_list = vocab['$'].tolist()

    diff = np.setdiff1d(random_train_unique_list, vocab_unique_list)
    new_vocab = list(set(random_train_unique_list) | set(vocab_unique_list))
    df = pd.DataFrame(new_vocab)
    df.to_csv('./data/vocab/vocab_random_train_decimal_2.csv', index=False)



def main():
    check_discrete()

if __name__ == "__main__":
    main()