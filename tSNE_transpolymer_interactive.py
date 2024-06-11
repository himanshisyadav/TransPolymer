import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import PandasTools
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs

from IPython.display import SVG
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.PandasTools import ChangeMoleculeRendering

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.nn import MaxPool1d

from transformers import AdamW, get_linear_schedule_with_warmup, RobertaModel, RobertaConfig, RobertaTokenizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import StandardScaler
from openTSNE import TSNE

import torchmetrics
from torchmetrics import R2Score

from PolymerSmilesTokenization import PolymerSmilesTokenizer
from dataset import Dataset_Emb, TransPolymerEmbeddings

#Bokeh library for plotting
import json
from bokeh.plotting import figure, show, output_notebook, ColumnDataSource
from bokeh.models import HoverTool, ColorBar
from bokeh.transform import factor_cmap
from bokeh.plotting import figure, output_file, save
from bokeh.transform import linear_cmap
from bokeh.palettes import Pastel1, Turbo256

output_notebook()

import colorsys 
from matplotlib import colors as mc

import pdb

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input: color (tuple) in RGB format, amount (float) by which to lighten the color. 
    Returns: (tuple) representing the lightened color in RGB format.
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def _prepareMol(mol,kekulize):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    return mc

def moltosvg(mol,molSize=(450,200),kekulize=True,drawer=None,**kwargs):
    mc = _prepareMol(mol,kekulize)
    if drawer is None:
        drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    drawer.DrawMolecule(mc,**kwargs)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return SVG(svg.replace('svg:',''))

def emb_convert(file_path, tokenizer, config):
    data = pd.read_csv(file_path)
    dataset = Dataset_Emb(data, tokenizer, tsne_config['blocksize'], config)
    dataloader = DataLoader(dataset, tsne_config['batch_size'], shuffle=False, num_workers=0)
    for step, batch in tqdm(enumerate(dataloader)):
        batch = batch.to(device)
        embeddings = batch.squeeze()
        embeddings = torch.transpose(embeddings, dim0=1, dim1=2)
        max_pool = MaxPool1d(kernel_size=tsne_config['blocksize'], padding=0,
                             dilation=1)  # Apply max pooling for conversion into t-SNE input
        embeddings = max_pool(embeddings)
        embeddings = torch.transpose(embeddings, dim0=1, dim1=2).reshape(embeddings.shape[0],
                                                                         768).cpu().detach().numpy()
        if step == 0:
            print("shape of embedding:", embeddings.shape)
            embeddings_all = embeddings
        else:
            embeddings_all = np.vstack((embeddings_all, embeddings))
    return  embeddings_all

def main(tsne_config):

    tokenizer = PolymerSmilesTokenizer.from_pretrained("/project/rcc/hyadav/roberta-base", max_len=tsne_config['blocksize'])
    config = RobertaConfig(
            vocab_size=50265,
            max_position_embeddings=514,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
    
    freqI_data_train = emb_convert(tsne_config['freqI_path_train'], tokenizer, config)
    # freqI_data_test = emb_convert(tsne_config['freqI_path_test'], tokenizer, config)
    print("start fitting t-SNE")
    tSNE = TSNE(
        perplexity=tsne_config['perplexity'],
        metric=tsne_config['metric'],
        n_jobs=tsne_config['n_jobs'],
        verbose=tsne_config['verbose'],
    )
    tSNE = tSNE.fit(freqI_data_train)
    # tSNE = tSNE.fit(freqI_data_test)
    print("finish fitting")

    df = pd.read_csv('./data/freqI_train_multi_comp_comb.csv')

    PandasTools.AddMoleculeColumnToFrame(df,'solv_comb_sm', 'solv')
    PandasTools.AddMoleculeColumnToFrame(df,'salt_sm', 'salt')

    svgs_salt = [moltosvg(m).data for m in df.salt]
    svgs_solv = [moltosvg(m).data for m in df.solv]

    ChangeMoleculeRendering(renderer='PNG')

    conductivity_values = df['conductivity_log'].values.tolist()

    source = ColumnDataSource(data=dict(x = tSNE[:,0], y = tSNE[:,1], svgs_salt=svgs_salt, svgs_solv=svgs_solv, desc= conductivity_values))

    hover = HoverTool(tooltips="""
        <div>
            <div> 
                <span style="font-size: 17px; font-weight: bold;"> Salt @svgs_salt{safe} </span>
            </div>
            <div> 
                <span style="font-size: 17px; font-weight: bold;"> Solvent @svgs_solv{safe} </span>
            </div>
            <div>
                <span style="font-size: 17px; font-weight: bold;"> Conductivity @desc </span>
            </div>
        </div>
        """
    )
    interactive_map = figure(width=1000, height=1000, tools=['reset,box_zoom,wheel_zoom,zoom_in,zoom_out,pan',hover])

    interactive_map.title.text = "Liquid Electrolytes (LLM)"
    interactive_map.title.align = "center"
    interactive_map.title.text_color = "orange"
    interactive_map.title.text_font_size = "25px"

    # # Original color in RGB format
    # original_color = (0, 0.576, 0.902)

    # # Generate a palette of 10 colors from the original color to its lighter shades
    # num_colors = 256
    # palette_lab = [mc.to_hex(lighten_color(original_color, amount=i/(num_colors-1))) for i in range(num_colors)]

    # # Print the palette for verification
    # print(palette_lab)

    #Use the field name of the column source
    mapper = linear_cmap(field_name = 'desc' , palette=Pastel1[9] ,low=min(conductivity_values) ,high=max(conductivity_values))

    interactive_map.circle('x', 'y', line_color=mapper, color=mapper, size=6, source=source, fill_alpha=0.2)
    color_bar = ColorBar(color_mapper=mapper['transform'], width=8)
    interactive_map.add_layout(color_bar, 'right')
    output_file("interactive_map_freqI_train_pastel9.html")
    save(interactive_map)
    
if __name__ == "__main__":

    tsne_config = yaml.load(open("config_tSNE.yaml", "r"), Loader=yaml.FullLoader)

    """Device"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(tsne_config)

