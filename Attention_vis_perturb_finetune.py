import pandas as pd
import numpy as np
import sys
import yaml

import torch
import torch.nn as nn
import json
import logging

from copy import deepcopy

from transformers import AdamW, get_linear_schedule_with_warmup, RobertaModel, RobertaConfig, RobertaTokenizer
from PolymerSmilesTokenization import PolymerSmilesTokenizer
import pdb
from Interpreter import Interpreter

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

        self.Regressor = nn.Sequential(
            nn.Dropout(drop_rate),
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
        fused = text_output * numeric_output

        #Regression 
        output = self.Regressor(fused)
        return output
    
def main(attention_config):
    data = pd.read_csv(attention_config['file_path'])
    smiles = data.values[attention_config['index'],0]

    if attention_config['add_vocab_flag']:
        vocab_sup = pd.read_csv(attention_config['vocab_sup_file'], header=None).values.flatten().tolist()
        tokenizer.add_tokens(vocab_sup)

    encoding = tokenizer(
        str(smiles),
        add_special_tokens=True,
        max_length=attention_config['blocksize'],
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    model = DownstreamRegression(drop_rate=0).to(device)
    checkpoint = torch.load(attention_config['model_path'])
    model = model.double()

    model.eval()
    with torch.no_grad():
        outputs = model.PretrainedModel(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)

    pdb.set_trace()



if __name__ == "__main__":

    attention_config = yaml.load(open("config_attention.yaml", "r"), Loader=yaml.FullLoader)

    """Device"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PretrainedModel = RobertaModel.from_pretrained(attention_config['pretrain_path']).to(device)
    tokenizer = PolymerSmilesTokenizer.from_pretrained("/project/rcc/hyadav/roberta-base", max_len=attention_config['blocksize'])

    main()