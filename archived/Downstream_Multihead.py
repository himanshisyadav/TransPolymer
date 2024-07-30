import pdb
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

np.random.seed(seed=1)
import torchinfo

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

        self.dropout = nn.Dropout(drop_rate)

        #Multihead Attention
        self.num_heads = 8
        self.hidden_dim = 256

        self.text_input_dim = self.PretrainedModel.config.hidden_size
        self.numeric_input_dim = 64

        self.numeric_input_dim_original = 1

        self.text_attention = nn.MultiheadAttention(embed_dim=self.text_input_dim, num_heads=self.num_heads)
        self.numeric_attention = nn.MultiheadAttention(embed_dim=self.numeric_input_dim, num_heads=self.num_heads)

        #Linear layer to change temp from dim 1 to dim 64
        self.linear_numeric_dim_change = nn.Linear(self.numeric_input_dim_original, self.numeric_input_dim)

        # Linear layers for combining the outputs of text and numeric attention
        self.linear_text = nn.Linear(self.text_input_dim, self.hidden_dim)
        self.linear_numeric = nn.Linear(self.numeric_input_dim, self.hidden_dim)
        self.linear_combine = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        
        self.Regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, input_ids, attention_mask, temp):
        outputs = self.PretrainedModel(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.last_hidden_state.permute(1,0,2)

        temp = temp.view(1, temp.shape[0], -1)

        text_input = logits
        numeric_input = self.linear_numeric_dim_change(temp.double())

        ## Multihead Attention Code
        # Apply attention mechanism to text input
        text_output, _ = self.text_attention(text_input, text_input, text_input)
        text_output = self.linear_text(text_output)

        # Apply attention mechanism to numeric input
        numeric_output, _ = self.numeric_attention(numeric_input, numeric_input, numeric_input)
        numeric_output = self.linear_numeric(numeric_output)

        # Concatenate and combine the outputs of text and numeric attention after making sequence lengths the same
        text_output = text_output[0, : , :] # 411 by 16 by 256
        numeric_output = numeric_output.squeeze(dim=0) # 1 by 16 by 256
        combined_output = torch.cat((text_output, numeric_output), dim=-1) #16 by 256
        fused = self.dropout(self.linear_combine(combined_output))

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
        # torchinfo.summary(model, verbose = 1)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, temp).float()
        loss = loss_fn(outputs.squeeze(), prop.squeeze())
        loss.backward()
        optimizer.step()
        scheduler.step()

    return None

def test(model, loss_fn, train_dataloader, test_dataloader, device, scaler, optimizer, scheduler, epoch):

    r2score = R2Score()
    train_loss = 0
    test_loss = 0
    # count = 0
    model.eval()
    with torch.no_grad():
        train_pred, train_true, test_pred, test_true = torch.tensor([]), torch.tensor([]), torch.tensor(
            []), torch.tensor([])

        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prop = batch["prop"].to(device).float()
            temp = batch["temp"].to(device).float()
            outputs = model(input_ids, attention_mask, temp).float()
            outputs = torch.from_numpy(scaler.inverse_transform(outputs.cpu().reshape(-1, 1)))
            prop = torch.from_numpy(scaler.inverse_transform(prop.cpu().reshape(-1, 1)))
            loss = loss_fn(outputs.squeeze(), prop.squeeze())
            train_loss += loss.item() * len(prop)
            # print("Train Loss: ", train_loss)
            train_pred = torch.cat([train_pred.to(device), outputs.to(device)])
            train_true = torch.cat([train_true.to(device), prop.to(device)])

        train_loss = train_loss / len(train_pred.flatten())
        r2_train = r2score(train_pred.flatten().to("cpu"), train_true.flatten().to("cpu")).item()
        print("train RMSE = ", np.sqrt(train_loss))
        print("train r^2 = ", r2_train)

        for step, batch in enumerate(test_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prop = batch["prop"].to(device).float()
            temp = batch["temp"].to(device).float()
            outputs = model(input_ids, attention_mask, temp).float()
            outputs = torch.from_numpy(scaler.inverse_transform(outputs.cpu().reshape(-1, 1)))
            prop = torch.from_numpy(scaler.inverse_transform(prop.cpu().reshape(-1, 1)))
            loss = loss_fn(outputs.squeeze(), prop.squeeze())
            test_loss += loss.item() * len(prop)
            test_pred = torch.cat([test_pred.to(device), outputs.to(device)])
            test_true = torch.cat([test_true.to(device), prop.to(device)])

        test_loss = test_loss / len(test_pred.flatten())
        r2_test = r2score(test_pred.flatten().to("cpu"), test_true.flatten().to("cpu")).item()
        print("test RMSE = ", np.sqrt(test_loss))
        print("test r^2 = ", r2_test)

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("r^2/train", r2_train, epoch)
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("r^2/test", r2_test, epoch)

    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'epoch': epoch}
    torch.save(state, finetune_config['save_path'])

    

    return train_loss, test_loss, r2_train, r2_test

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

            train_dataset = Downstream_Dataset(train_data, tokenizer, finetune_config['blocksize'])
            test_dataset = Downstream_Dataset(test_data, tokenizer, finetune_config['blocksize'])
            train_dataloader = DataLoader(train_dataset, finetune_config['batch_size'], shuffle=True, num_workers=finetune_config["num_workers"])
            test_dataloader = DataLoader(test_dataset, finetune_config['batch_size'], shuffle=False, num_workers=finetune_config["num_workers"])

            """Parameters for scheduler"""
            steps_per_epoch = train_data.shape[0] // finetune_config['batch_size']
            training_steps = steps_per_epoch * finetune_config['num_epochs']
            warmup_steps = int(training_steps * finetune_config['warmup_ratio'])

            """Train the model"""
            model = DownstreamRegression(drop_rate=finetune_config['drop_rate']).to(device)
            model = model.double()
            # loss_fn = nn.MSELoss()
            loss_fn = nn.HuberLoss(delta = 5)

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
            train_loss_best, test_loss_best, best_train_r2, best_test_r2 = 0.0, 0.0, 0.0, 0.0  # Keep track of the best test r^2 in one fold. If cross-validation is not used, that will be the same as best_r2.
            count = 0     # Keep track of how many successive non-improvement epochs
            for epoch in range(finetune_config['num_epochs']):
                print("epoch: %s/%s" % (epoch+1, finetune_config['num_epochs']))
                train(model, optimizer, scheduler, loss_fn, train_dataloader, device)
                train_loss, test_loss, r2_train, r2_test = test(model, loss_fn, train_dataloader,
                                                                                   test_dataloader, device, scaler,
                                                                                   optimizer, scheduler, epoch)
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
                    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch, 'fold:': fold}
                    torch.save(state, finetune_config['best_model_path'])         # save the best model

                if count >= finetune_config['tolerance']:
                    print("Early stop")
                    if best_test_r2 == 0:
                        print("Poor performance with negative r^2")
                    break

            train_loss_avg.append(np.sqrt(train_loss_best))
            test_loss_avg.append(np.sqrt(test_loss_best))
            train_r2_avg.append(best_train_r2)
            test_r2_avg.append(best_test_r2)
            writer.flush()

        """Average of metrics over all folds"""
        train_rmse = np.mean(np.array(train_loss_avg))
        test_rmse = np.mean(np.array(test_loss_avg))
        train_r2 = np.mean(np.array(train_r2_avg))
        test_r2 = np.mean(np.array(test_r2_avg))
        std_test_rmse = np.std(np.array(test_loss_avg))
        std_test_r2 = np.std(np.array(test_r2_avg))

        print("Train RMSE =", train_rmse)
        print("Test RMSE =", test_rmse)
        print("Train R^2 =", train_r2)
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

        """Train the model"""
        model = DownstreamRegression(drop_rate=finetune_config['drop_rate']).to(device)
        model = model.double()
        # loss_fn = nn.MSELoss()
        loss_fn = nn.HuberLoss(delta = 5)

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
        train_loss_best, test_loss_best, best_train_r2, best_test_r2 = 0.0, 0.0, 0.0, 0.0  # Keep track of the best test r^2 in one fold. If cross-validation is not used, that will be the same as best_r2.
        count = 0     # Keep track of how many successive non-improvement epochs
        for epoch in range(finetune_config['num_epochs']):
            print("epoch: %s/%s" % (epoch+1,finetune_config['num_epochs']))
            train(model, optimizer, scheduler, loss_fn, train_dataloader, device)
            train_loss, test_loss, r2_train, r2_test = test(model, loss_fn, train_dataloader,
                                                                                   test_dataloader, device, scaler,
                                                                                   optimizer, scheduler, epoch)
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
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch}
                torch.save(state, finetune_config['best_model_path'])         # save the best model

            if count >= finetune_config['tolerance']:
                print("Early stop")
                if best_test_r2 == 0:
                    print("Poor performance with negative r^2")
                break

        writer.flush()


if __name__ == "__main__":

    finetune_config = yaml.load(open("config_finetune.yaml", "r"), Loader=yaml.FullLoader)
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
        tokenizer = RobertaTokenizer.from_pretrained("/project/rcc/hyadav/roberta-base", max_len=finetune_config['blocksize'])
    max_token_len = finetune_config['blocksize']

    """Run the main function"""
    main(finetune_config)





