import pandas as pd
import numpy as np
import sys
import yaml

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm

from transformers import AdamW, get_linear_schedule_with_warmup, RobertaModel, RobertaConfig, RobertaTokenizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from rdkit import Chem

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

from packaging import version

import torchmetrics
from torchmetrics import R2Score

from PolymerSmilesTokenization import PolymerSmilesTokenizer
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

import pdb

from colorama import init, Fore, Style

import re
from collections import defaultdict

from highlight_text import ax_text

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
        self.numeric_featurizer = nn.Linear(1, self.PretrainedModel.config.hidden_size)

        self.Regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            # nn.Linear(self.PretrainedModel.config.hidden_size, self.PretrainedModel.config.hidden_size),
            # nn.SiLU(),
            nn.Linear(self.PretrainedModel.config.hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask, temp):
        outputs = self.PretrainedModel(input_ids=input_ids, attention_mask=attention_mask)
        
        # Global Average Pooling
        last_hidden_state = outputs.last_hidden_state[:,:,:]
        pooled_output = self.pooler(last_hidden_state)
        logits = pooled_output

        # Getting Temperature Values 
        temp = temp.reshape(-1, 1).float()

        # Fusion 3: Simple Linear Fusion
        text_input = logits
        numeric_input = temp

        # Process text input, convert to a feature vector of size pretrain hidden dim
        text_output = text_input
        
        # Process numeric input
        numeric_output = self.numeric_featurizer(numeric_input)
        
        # Compute fusion 
        # fused = (text_output + numeric_output) / 2
        # fused = torch.mean(torch.stack([text_output, numeric_output]), dim=0)
        # fused = torch.cat((text_output, numeric_output), 1)
        fused = text_output * numeric_output

        #Regression 
        output = self.Regressor(fused)
        return output

def clip_number(match):
    number = float(match.group())
    return f"{number:.2f}"

def main(attention_config):
    if attention_config['task'] == 'pretrain':
        # smiles = attention_config['smiles']
        data = pd.read_csv(attention_config['file_path'])
        smiles = data.values[attention_config['index'],0]
    else:
        data = pd.read_csv(attention_config['file_path'])
        smiles = data.values[attention_config['index'],0]
        # temp = data.values[attention_config['index'],1]
        # pdb.set_trace()
    
    # Regular expression to find the numbers in the input string
    pattern = r"[-]?\d+\.\d+"

    smiles = re.sub(pattern, clip_number, smiles)

    print("SMILES length", len(smiles))

    if attention_config['add_vocab_flag']:
        vocab_sup = pd.read_csv(attention_config['vocab_sup_file'], header=None).values.flatten().tolist()
        tokenizer.add_tokens(vocab_sup)

    encoding = tokenizer(
        str(smiles),
        add_special_tokens=True,
        max_length=len(smiles),
        return_token_type_ids=False,
        # padding="max_length",
        padding = 'do_not_pad',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    # temp = encoding["temp"].to(device)

    if attention_config['task'] == 'pretrain':
        outputs = PretrainedModel(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
    else:
        model = DownstreamRegression(drop_rate=0).to(device)
        checkpoint = torch.load(attention_config['model_path'])
        # model.load_state_dict(checkpoint['model'])
        model = model.double()

        model.eval()
        with torch.no_grad():
            outputs = model.PretrainedModel(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)

    attention = outputs[-1]
    xticklabels = tokenizer.convert_ids_to_tokens(input_ids.squeeze())
    xticklabels = [token for token in xticklabels if token != '<pad>']

    print(smiles)
    print(xticklabels)

    fig, axes = plt.subplots(3,4, figsize=(attention_config['figsize_x'],attention_config['figsize_y']))
    # fig.subplots_adjust(hspace=0.4, wspace=0.4) 
    
    if attention_config['task'] == 'pretrain':
        for i in range(3):
            for j in range(4):
                sns.heatmap(attention[attention_config['layer']][0,4*i+j,:,:].cpu().detach().numpy(), ax = axes[i,j], xticklabels=xticklabels, yticklabels=xticklabels,cbar_kws={'shrink': 0.7})
                axes[i,j].set_title(label="Attention Head %s" % str(4*i+j+1), fontsize=attention_config['fontsize'])
                axes[i,j].tick_params(labelsize=attention_config['labelsize'])

                # axes[i, j].set_xticklabels(xticklabels, rotation=attention_config['rotation'], ha='right')
                # axes[i, j].set_yticklabels(xticklabels, rotation=0, va='center')

                # Adjust x-axis (vertical tokens)
                x_labels = [token if len(token) == 1 else f'{token}' for token in xticklabels]
                # pdb.set_trace()
                axes[i,j].set_xticks(list(range(len(x_labels))))
                axes[i,j].set_xticklabels(x_labels, fontsize=attention_config['labelsize'], rotation=attention_config['rotation_x'], ha='center')

                # Adjust y-axis (horizontal tokens for >1 character)
                y_labels = [token if len(token) == 1 else f'{token}' for token in xticklabels]
                axes[i,j].set_yticks(list(range(len(y_labels))))
                axes[i,j].set_yticklabels(y_labels, fontsize=attention_config['labelsize'], rotation=attention_config['rotation_y'], ha='right')

                cbar = axes[i,j].collections[0].colorbar
                # here set the labelsize by 20
                cbar.ax.tick_params(labelsize=attention_config['labelsize'])
        fig.suptitle(f'Pretrain Maps for {smiles}', fontsize=attention_config['fontsize'], y=1.02)
        plt.savefig(attention_config['save_path'], bbox_inches='tight')
    
    else:
        # Find the index of the last $
        # last_dollar_index = len(xticklabels) - 1 - xticklabels[::-1].index('$')

        # # Get the indices after the last $ and before </s>
        # indices = list(range(last_dollar_index + 1, xticklabels.index('</s>')))
        # # Extract elements at the specified indices
        # salt_tokens = [xticklabels[i] for i in indices]

        # print("Salt Tokens: ", salt_tokens)

        ##### Getting indices of everything except for the 2nd,3rd,4th solvents
        # # Get the index of the first | element
        # first_pipe_index = xticklabels.index('|')

        # # Get the index of the 4th | element
        # fourth_pipe_index = [i for i, x in enumerate(xticklabels) if x == '|'][3]

        # # Indices before the first | element
        # indices_before_first_pipe = list(range(first_pipe_index))

        # # Indices after the 4th | element
        # indices_after_fourth_pipe = list(range(fourth_pipe_index + 1, len(xticklabels)))

        # indices = indices_before_first_pipe + indices_after_fourth_pipe

        # print("Tokens in Figure", [xticklabels[i] for i in indices])

        most_attended = []

        yticklabels = ['Layer 1','Layer 2','Layer 3','Layer 4','Layer 5','Layer 6']
        for i in range(3):
            for j in range(4):
                for layer in range(6):
                    attention_sub = attention[layer][0,4*i+j,0,:].cpu().detach().numpy().reshape(1,-1)

                    # Find the most attended token
                    # pdb.set_trace()
                    max_idx = torch.argmax(attention[layer][0,4*i+j,0,:]).item()
                    max_token = xticklabels[max_idx] if max_idx < len(xticklabels) else f"Token {max_idx}"
                    max_score = attention[layer][0,4*i+j,0,max_idx].item()
                    
                    # Print the result
                    # print(f"  Head {4*i+j + 1}: Most attended token = '{max_token}', Attention score = {max_score:.4f}")
                    
                    # Append to the result list
                    most_attended.append({
                        "layer": layer + 1,
                        "head": 4*i+j + 1,
                        "token": max_token,
                        "score": max_score,
                        "index": max_idx
                    })

                    if layer == 0:
                        attention_CLS = attention_sub
                    else:
                        attention_CLS = np.vstack((attention_CLS, attention_sub))
                max_attention_CLS = attention_CLS.max(axis = 0)
                # max_attention= np.vstack(max_attention_CLS.reshape(1,max_attention_CLS.shape[0]))

                sns.heatmap(attention_CLS, ax = axes[i,j], xticklabels=False)
                axes[i, j].set_title(label="Attention Head %s" % str(4 * i + j + 1), fontsize=attention_config['fontsize'])
                axes[i, j].set_yticklabels(rotation=attention_config['rotation'], labels=yticklabels)
                axes[i, j].tick_params(labelsize=attention_config['labelsize'])
                cbar = axes[i, j].collections[0].colorbar
                # here set the labelsize by 20
                cbar.ax.tick_params(labelsize=attention_config['labelsize'])
        # fig.suptitle(f'Attention Scores for {smiles}', fontsize=attention_config['fontsize'], y=1.02)

        # Sum the scores for common tokens
        token_scores = defaultdict(float)
        for entry in most_attended:
            token_scores[entry["index"]] += entry["score"]

        # Get the top 5 token names by score
        top_5_token_indices = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)[:5]

        highlight_indices = [idx for idx, _ in top_5_token_indices]

        for idx in highlight_indices:
            print("Top Tokens", xticklabels[idx])
        
        bbox_left = axes[0,0].get_position()
        title_start = bbox_left.x0 + 0.001
        bbox_right = axes[0,3].get_position()
        title_end = bbox_right.x1 - 0.001

        # print(title_start, title_end)

        x_pos = title_start

        full_text = "".join(xticklabels)
        base_spacing = ((title_end - title_start) / len(full_text))
        spacing = base_spacing
        # print(spacing)

        highlight_color = 'red'
        for idx, token in enumerate(xticklabels):
            if len(token) > 2:
                spacing = base_spacing + base_spacing * (len(token)-1.5) 
            else:   
                spacing = base_spacing + base_spacing * (len(token)-1)  
            if idx in highlight_indices:
                color = highlight_color
            else:
                color = 'black'
            # print(token, x_pos, spacing)
            plt.text(x = x_pos, y = 0.92, s = token, color=color, fontsize=attention_config["fontsize"], transform=fig.transFigure)
            x_pos += spacing  # Adjust spacing between characters

        fig.tight_layout()  
        fig.suptitle("Attention Map", fontsize=attention_config["fontsize"])  
        save_path = f"{attention_config['save_path']}_index_{attention_config['index']}_run_{attention_config['run_id']}.png"
        print(save_path)
        plt.savefig(save_path, bbox_inches='tight')

if __name__ == "__main__":

    attention_config = yaml.load(open("config_attention.yaml", "r"), Loader=yaml.FullLoader)

    """Device"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PretrainedModel = RobertaModel.from_pretrained(attention_config['pretrain_path']).to(device)
    tokenizer = PolymerSmilesTokenizer.from_pretrained("/project/rcc/hyadav/roberta-base", max_len=attention_config['blocksize'])

    main(attention_config)

