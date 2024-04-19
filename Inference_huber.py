import pandas as pd
import numpy as np
import sys
import yaml

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm
import torch.nn.functional as F

from transformers import AdamW, get_linear_schedule_with_warmup, RobertaModel, RobertaConfig, RobertaTokenizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from rdkit import Chem

from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

from packaging import version

import torchmetrics
from torchmetrics import R2Score

from PolymerSmilesTokenization import PolymerSmilesTokenizer
from dataset import Downstream_Dataset, DataAugmentation

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from copy import deepcopy

import pdb
import seaborn as sns
from sklearn.metrics import mean_absolute_error as mae 

np.random.seed(seed=1)

"""Layer-wise learning rate decay"""

def roberta_base_AdamW_LLRD(model, lr, weight_decay):
    opt_parameters = []  # To be passed to the optimizer (only parameters of the layers you want to update).
    named_parameters = list(model.named_parameters())
    print("number of named parameters =", len(named_parameters))

    # According to AAAMLP book by A. Thakur, we generally do not use any decay
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # === Pooler and Regressor ======================================================

    params_0 = [p for n, p in named_parameters if ("pooler" in n or "Regressor" in n)
                and any(nd in n for nd in no_decay)]
    print("params in pooler and regressor without decay =", len(params_0))
    params_1 = [p for n, p in named_parameters if ("pooler" in n or "Regressor" in n)
                and not any(nd in n for nd in no_decay)]
    print("params in pooler and regressor with decay =", len(params_1))

    head_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
    opt_parameters.append(head_params)

    head_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
    opt_parameters.append(head_params)

    print("pooler and regressor lr =", lr)

    # === Hidden layers ==========================================================

    for layer in range(5, -1, -1):
        params_0 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n
                    and any(nd in n for nd in no_decay)]
        print(f"params in hidden layer {layer} without decay =", len(params_0))
        params_1 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n
                    and not any(nd in n for nd in no_decay)]
        print(f"params in hidden layer {layer} with decay =", len(params_1))

        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        opt_parameters.append(layer_params)

        layer_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
        opt_parameters.append(layer_params)

        print("hidden layer", layer, "lr =", lr)

        lr *= 0.9

        # === Embeddings layer ==========================================================

    params_0 = [p for n, p in named_parameters if "embeddings" in n
                and any(nd in n for nd in no_decay)]
    print("params in embeddings layer without decay =", len(params_0))
    params_1 = [p for n, p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay)]
    print("params in embeddings layer with decay =", len(params_1))

    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
    opt_parameters.append(embed_params)

    embed_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
    opt_parameters.append(embed_params)
    print("embedding layer lr =", lr)

    return AdamW(opt_parameters, lr=lr)

"""Model"""
class GlobalAveragePooling1D(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling1D, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=1)

class DownstreamRegression(nn.Module):
    def __init__(self, drop_rate=0.1):
        super(DownstreamRegression, self).__init__()
        self.PretrainedModel = deepcopy(PretrainedModel)
        self.PretrainedModel.resize_token_embeddings(len(tokenizer))
        self.pooler = GlobalAveragePooling1D()

        self.attention_dim = 200
        self.constant_dim = 1
        self.embedding_dim = self.PretrainedModel.config.hidden_size

        #Attention 
        # self.fc_embed = nn.Linear(self.embedding_dim, self.attention_dim)
        # self.fc_const = nn.Linear(self.constant_dim, self.attention_dim)
        # self.fc_out = nn.Linear(self.attention_dim, self.embedding_dim + self.constant_dim)

        #Fusion        
        # self.fusion = torch.nn.Linear(self.PretrainedModel.config.hidden_size + self.constant_dim, self.PretrainedModel.config.hidden_size + self.constant_dim)
        # self.relu = torch.nn.ReLU()
        # self.dropout = torch.nn.Dropout(drop_rate)

        #Fusion 2
        self.hidden_dim = 256
        self.linear_text = nn.Linear(self.PretrainedModel.config.hidden_size, self.hidden_dim)
        self.linear_numeric = nn.Linear(self.constant_dim, self.hidden_dim)

        self.output_layer = nn.Linear(self.hidden_dim, self.PretrainedModel.config.hidden_size + self.constant_dim) 
        
        self.Regressor = nn.Sequential(
        #     nn.Dropout(drop_rate),
        #     nn.Linear(self.PretrainedModel.config.hidden_size + self.constant_dim, self.PretrainedModel.config.hidden_size + self.constant_dim),
        #     nn.SiLU(),
        #     nn.Linear(self.PretrainedModel.config.hidden_size + self.constant_dim, 1)
        # )
            nn.Dropout(drop_rate),
            nn.Linear(self.PretrainedModel.config.hidden_size + self.constant_dim, 1),
        )
        #     nn.Dropout(drop_rate),
        #     nn.Linear(self.PretrainedModel.config.hidden_size + self.constant_dim, 128),
        #     nn.Dropout(drop_rate),
        #     nn.Linear(128, 64),
        #     nn.Dropout(drop_rate),
        #     nn.Linear(64, 1)
        # )

    def forward(self, input_ids, attention_mask, temp):
        outputs = self.PretrainedModel(input_ids=input_ids, attention_mask=attention_mask)
        # Using <s> token
        # logits = outputs.last_hidden_state[:, 0, :]
        
        # Global Average Pooling
        last_hidden_state = outputs.last_hidden_state[:,:,:]
        pooled_output = self.pooler(last_hidden_state)
        logits = pooled_output

        # Getting Temperature Values 
        temp = temp.reshape(-1, 1).float()

        ## Attention Code
        # embedding = logits.double() # 16 by 768
        # constant = temp.double() # 16 by 1

        # embed_attention = F.softmax(torch.tanh(self.fc_embed(embedding)), dim=1) # 16 by 200
        # const_attention = F.softmax(torch.tanh(self.fc_const(constant)), dim=1) # 16 by 200
        
        # # Weighted sum of embeddings and constant values
        # embedding= embedding[:, :self.attention_dim]
        # embed_weighted = embedding * embed_attention # 16 by 768 mul 16 by 200
        # const_weighted = constant * const_attention # 16 by 1 mul 16 by 200

        # combined = embed_weighted + const_weighted
        # fused = self.fc_out(combined)

        ## Non-attention Code
        # concated_data = torch.cat((logits, temp), dim=1)
        # fused = self.dropout(self.relu(self.fusion(concated_data)))

        ##Fusion 2
        # text_input = logits.double()
        # numeric_input = temp.double()
        # # Process text input
        # text_output = F.relu(self.linear_text(text_input))
        
        # # Process numeric input
        # numeric_output = F.relu(self.linear_numeric(numeric_input))
        
        # # Compute mean fusion
        # fusion_output = (text_output + numeric_output) / 2
        # fused = self.output_layer(fusion_output)

        # Fusion 3: Simple Linear Fusion
        text_input = logits
        numeric_input = temp
        # Process text input
        text_output = self.linear_text(text_input)
        
        # Process numeric input
        numeric_output = self.linear_numeric(numeric_input)
        
        # Compute mean fusion
        # fusion_output = (text_output + numeric_output) / 2
        # fused = self.output_layer(fusion_output)

        #Max Fusion
        fused = self.output_layer(torch.maximum(text_output,numeric_output))

        #Regression 
        output = self.Regressor(fused)
        return output

"""Train"""

def train(model, optimizer, scheduler, loss_fn, train_dataloader, device):

    model.train()

    print("Number of parameteres are: ", sum(param.numel() for param in model.parameters() if param.requires_grad)) 

    for step, batch in enumerate(train_dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        temp = batch["temp"].to(device).float()
        prop = batch["prop"].to(device).float()
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, temp).float()
        loss = loss_fn(outputs.squeeze(), prop.squeeze())
        loss.backward()
        optimizer.step()
        scheduler.step()

    return None

def test(model, loss_fn, train_dataloader, test_dataloader, device, optimizer, scheduler, scaler, epoch):

    r2score = R2Score()
    test_loss = 0
    # count = 0
    model.eval()
    with torch.no_grad():
        test_pred, test_true = torch.tensor([]), torch.tensor([])

        for step, batch in enumerate(test_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            temp = batch["temp"].to(device).float()
            prop = batch["prop"].to(device).float()
            outputs = model(input_ids, attention_mask, temp).float()
            outputs = torch.from_numpy(scaler.inverse_transform(outputs.cpu().reshape(-1, 1)))
            prop = torch.from_numpy(scaler.inverse_transform(prop.cpu().reshape(-1, 1)))
            loss = loss_fn(outputs.squeeze(), prop.squeeze())
            test_loss += loss.item() * len(prop)
            test_pred = torch.cat([test_pred.to(device), outputs.to(device)])
            test_true = torch.cat([test_true.to(device), prop.to(device)])

        # pdb.set_trace()

        test_loss = test_loss / len(test_pred.flatten())
        r2_test = r2score(test_pred.flatten().to("cpu"), test_true.flatten().to("cpu")).item()
        mae_error_test = mae(test_true.flatten().to("cpu"), test_pred.flatten().to("cpu")) 
        print("test RMSE = ", np.sqrt(test_loss))
        print("test r^2 = ", r2_test)
        print("test MAE =", mae_error_test)

    # Inference Plot

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(test_true.flatten().to("cpu"),test_pred.flatten().to("cpu"), 'o', color = 'green', markersize = '1')
    xl, xr = ax.get_xlim()
    yt, yb = ax.get_ylim()
    left = xl + 0.5
    top = yt + 0.5

    # Add diagonal line
    min_val = min(torch.min(test_true.flatten().to("cpu")), torch.min(test_pred.flatten().to("cpu")))
    max_val = max(torch.max(test_true.flatten().to("cpu")), torch.max(test_pred.flatten().to("cpu")))
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Diagonal Line')

    ax.text(left,top, 'RMSE=' + str(round(np.sqrt(test_loss),3)) + ' R2=' + str(round(r2_test,3)) + ' MAE=' + str(round(mae_error_test,3)), ha='left', va='top')
    plt.grid(True)
    file_name = "./plots/inference_plot_rmse_" + str(np.sqrt(test_loss)) + "_r2_" + str(r2_test) + "_mae_" + str(mae_error_test) + ".png"
    plt.savefig(file_name)

    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("r^2/test", r2_test, epoch)

    # pdb.set_trace()

    return test_loss, r2_test

    """

    if r2_test > best_test_r2:
        best_train_r2 = r2_train
        best_test_r2 = r2_test
        train_loss_best = train_loss
        test_loss_best = test_loss
        count = 0
    else:
        count += 1

    if r2_test > best_r2:
        best_r2 = r2_test
        torch.save(state, finetune_config['best_model_path'])         # save the best model

    if count >= finetune_config['tolerance']:
        print("Early stop")
        if best_test_r2 == 0:
            print("Poor performance with negative r^2")
            return None
        else:
            return train_loss_best, test_loss_best, best_train_r2, best_test_r2, best_r2

    return train_loss_best, test_loss_best, best_train_r2, best_test_r2, best_r2
    """

def main(finetune_config):

    """Tokenizer"""
    if finetune_config['add_vocab_flag']:
        vocab_sup = pd.read_csv(finetune_config['vocab_sup_file'], header=None).values.flatten().tolist()
        tokenizer.add_tokens(vocab_sup)

    best_r2 = 0.0           # monitor the best r^2 in the run

    """Data"""
    if finetune_config['CV_flag']:
        print("Start Cross Validation")
        data = pd.read_csv(finetune_config['train_file'])
        """K-fold"""
        splits = KFold(n_splits=finetune_config['k'], shuffle=True,
                       random_state=1)  # k=1 for train-test split and k=5 for cross validation
        train_loss_avg, test_loss_avg, train_r2_avg, test_r2_avg = [], [], [], []     # monitor the best metrics in each fold
        for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(data.shape[0]))):
            print('Fold {}'.format(fold + 1))

            train_data = data.loc[train_idx, :].reset_index(drop=True)
            test_data = data.loc[val_idx, :].reset_index(drop=True)

            if finetune_config['aug_flag']:
                print("Data Augamentation")
                DataAug = DataAugmentation(finetune_config['aug_indicator'])
                train_data = DataAug.smiles_augmentation(train_data)
                if finetune_config['aug_special_flag']:
                    train_data = DataAug.smiles_augmentation_2(train_data)
                    train_data = DataAug.combine_smiles(train_data)
                    test_data = DataAug.combine_smiles(test_data)
                train_data = DataAug.combine_columns(train_data)
                test_data = DataAug.combine_columns(test_data)

            scaler = StandardScaler()
            train_data.iloc[:, 1] = scaler.fit_transform(train_data.iloc[:, 1].values.reshape(-1, 1))
            test_data.iloc[:, 1] = scaler.transform(test_data.iloc[:, 1].values.reshape(-1, 1))
            train_data.iloc[:, 2] = scaler.fit_transform(train_data.iloc[:, 2].values.reshape(-1, 1))
            test_data.iloc[:, 2] = scaler.transform(test_data.iloc[:, 2].values.reshape(-1, 1))

            train_dataset = Downstream_Dataset(train_data, tokenizer, finetune_config['blocksize'])
            test_dataset = Downstream_Dataset(test_data, tokenizer, finetune_config['blocksize'])
            train_dataloader = DataLoader(train_dataset, finetune_config['batch_size'], shuffle=True, num_workers=finetune_config["num_workers"])
            test_dataloader = DataLoader(test_dataset, finetune_config['batch_size'], shuffle=False, num_workers=finetune_config["num_workers"])

            """Parameters for scheduler"""
            steps_per_epoch = train_data.shape[0] // finetune_config['batch_size']
            training_steps = steps_per_epoch * finetune_config['num_epochs']
            warmup_steps = int(training_steps * finetune_config['warmup_ratio'])

            # """Train the model"""
            # model = DownstreamRegression(drop_rate=finetune_config['drop_rate']).to(device)
            # model = model.double()
            # loss_fn = nn.MSELoss()
            loss_fn = nn.HuberLoss(delta = 5)

            """Load the model"""
            model = DownstreamRegression(drop_rate=finetune_config['drop_rate']).to(device)
            model_dict = torch.load(finetune_config['best_model_path'], map_location='cpu')
            model.load_state_dict(model_dict['model'])
            optimizer = model_dict['optimizer']
            scheduler = model_dict['scheduler']


            if finetune_config['LLRD_flag']:
                optimizer = roberta_base_AdamW_LLRD(model, finetune_config['lr_rate'], finetune_config['weight_decay'])
            else:
                optimizer = AdamW(
                    [
                        {"params": model.PretrainedModel.parameters(), "lr": finetune_config['lr_rate'],
                         "weight_decay": 0.0},
                        {"params": model.Regressor.parameters(), "lr": finetune_config['lr_rate_reg'],
                         "weight_decay": finetune_config['weight_decay']},
                    ]
                )

            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=training_steps)
            torch.cuda.empty_cache()
            test_loss_best, best_test_r2 = 0.0, 0.0, 0.0, 0.0  # Keep track of the best test r^2 in one fold. If cross-validation is not used, that will be the same as best_r2.
            count = 0     # Keep track of how many successive non-improvement epochs
            for epoch in range(finetune_config['num_epochs']):
                print("epoch: %s/%s" % (epoch+1, finetune_config['num_epochs']))
                # train(model, optimizer, scheduler, loss_fn, train_dataloader, device)
                test_loss, r2_test = test(model, loss_fn, train_dataloader,
                                                                                   test_dataloader, device, scaler,
                                                                                   optimizer, scheduler, epoch)

            test_loss_avg.append(np.sqrt(test_loss_best))
            test_r2_avg.append(best_test_r2)
            writer.flush()

        """Average of metrics over all folds"""
        test_rmse = np.mean(np.array(test_loss_avg))
        test_r2 = np.mean(np.array(test_r2_avg))
        std_test_rmse = np.std(np.array(test_loss_avg))
        std_test_r2 = np.std(np.array(test_r2_avg))

        print("Test RMSE =", test_rmse)
        print("Test R^2 =", test_r2)
        print("Standard Deviation of Test RMSE =", std_test_rmse)
        print("Standard Deviation of Test R^2 =", std_test_r2)

    else:
        print("Train Test Split")
        train_data = pd.read_csv(finetune_config['train_file'])
        test_data = pd.read_csv(finetune_config['test_file'])

        if finetune_config['aug_flag']:
            print("Data Augmentation")
            DataAug = DataAugmentation(finetune_config['aug_indicator'])
            train_data = DataAug.smiles_augmentation(train_data)
            if finetune_config['aug_special_flag']:
                train_data = DataAug.smiles_augmentation_2(train_data)
                train_data = DataAug.combine_smiles(train_data)
                test_data = DataAug.combine_smiles(test_data)
            train_data = DataAug.combine_columns(train_data)
            test_data = DataAug.combine_columns(test_data)

        scaler = StandardScaler()
        train_data.iloc[:, 1] = scaler.fit_transform(train_data.iloc[:, 1].values.reshape(-1, 1))
        test_data.iloc[:, 1] = scaler.transform(test_data.iloc[:, 1].values.reshape(-1, 1))
        train_data.iloc[:, 2] = scaler.fit_transform(train_data.iloc[:, 2].values.reshape(-1, 1))
        test_data.iloc[:, 2] = scaler.transform(test_data.iloc[:, 2].values.reshape(-1, 1))

        train_dataset = Downstream_Dataset(train_data, tokenizer, finetune_config['blocksize'])
        test_dataset = Downstream_Dataset(test_data, tokenizer, finetune_config['blocksize'])
        train_dataloader = DataLoader(train_dataset, finetune_config['batch_size'], shuffle=True, num_workers=finetune_config["num_workers"])
        test_dataloader = DataLoader(test_dataset, finetune_config['batch_size'], shuffle=False, num_workers=finetune_config["num_workers"])

        """Parameters for scheduler"""
        steps_per_epoch = train_data.shape[0] // finetune_config['batch_size']
        training_steps = steps_per_epoch * finetune_config['num_epochs']
        warmup_steps = int(training_steps * finetune_config['warmup_ratio'])

        """Load the model"""
        model = DownstreamRegression(drop_rate=finetune_config['drop_rate']).to(device)
        model_dict = torch.load(finetune_config['best_model_path'], map_location='cpu')
        model.load_state_dict(model_dict['model'])
        optimizer = model_dict['optimizer']
        scheduler = model_dict['scheduler']
        # loss_fn = nn.MSELoss()
        loss_fn = nn.HuberLoss(delta = 5)

        # if finetune_config['LLRD_flag']:
        #     optimizer = roberta_base_AdamW_LLRD(model, finetune_config['lr_rate'], finetune_config['weight_decay'])
        # else:
        #     pdb.set_trace()
        #     optimizer = AdamW(
        #         [
        #             {"params": model.PretrainedModel.parameters(), "lr": finetune_config['lr_rate'],
        #              "weight_decay": 0.0},
        #             {"params": model.Regressor.parameters(), "lr": finetune_config['lr_rate_reg'],
        #              "weight_decay": finetune_config['weight_decay']},
        #         ]
        #     )

        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    # num_training_steps=training_steps)
        torch.cuda.empty_cache()
        test_loss_best, best_test_r2 = 0.0, 0.0  # Keep track of the best test r^2 in one fold. If cross-validation is not used, that will be the same as best_r2.
        count = 0     # Keep track of how many successive non-improvement epochs
        for epoch in range(finetune_config['num_epochs']):
            print("epoch: %s/%s" % (epoch+1,finetune_config['num_epochs']))
            # train(model, optimizer, scheduler, loss_fn, train_dataloader, device)
            test_loss, r2_test = test(model, loss_fn, train_dataloader, test_dataloader, device, optimizer, scheduler, scaler, epoch)

        writer.flush()


if __name__ == "__main__":

    finetune_config = yaml.load(open("config_inference.yaml", "r"), Loader=yaml.FullLoader)
    print(finetune_config)

    """Device"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    if finetune_config['model_indicator'] == 'pretrain':
        print("Use the pretrained model")
        PretrainedModel = RobertaModel.from_pretrained(finetune_config['model_path'])
        tokenizer = PolymerSmilesTokenizer.from_pretrained("/project/rcc/hyadav/roberta-base", max_len=finetune_config['blocksize'])
        PretrainedModel.config.hidden_dropout_prob = finetune_config['hidden_dropout_prob']
        PretrainedModel.config.attention_probs_dropout_prob = finetune_config['attention_probs_dropout_prob']
    else:
        print("No Pretrain")
        config = RobertaConfig(
            vocab_size=50265,
            max_position_embeddings=514,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        PretrainedModel = RobertaModel(config=config)
        tokenizer = RobertaTokenizer.from_pretrained("/project/rcc/hyadav/ChemBERTa-77M-MLM", max_len=finetune_config['blocksize'])
    max_token_len = finetune_config['blocksize']

    """Run the main function"""
    main(finetune_config)






