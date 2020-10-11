import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import os 
import re
import numpy as np 
import pandas as pd 
from PIL import Image 
from tokenizers import BertWordPieceTokenizer
from AttentionTransformer.TrainSeq import device 

from Transformer import Transformer
from torch.autograd import Variable 
import matplotlib.pyplot as plt 
from torchvision import transforms

SEED = 3007
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

data = pd.read_csv('../data/cocoImageCaptions1k.csv')
tokenizer = BertWordPieceTokenizer('../data/bert-word-piece-custom-wikitext-vocab-10k-vocab.txt', lowercase=True)

simple_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

## parameters
target_vocab_size = tokenizer.get_vocab_size()
target_pad_id = 0
emb_dim = 512
hid_dim = 512
dim_model = emb_dim 
dim_inner = emb_dim * 4 
layers = 4
heads = 6
dim_key = 64 
dim_value = 64 
dropout = 0.1
seq_len = 100

model = Transformer(
    target_vocab_size = target_vocab_size, target_pad_id = target_pad_id,
    emb_dim = emb_dim, hid_dim = hid_dim, dim_model = dim_model, 
    dim_inner = dim_inner, layers = layers, heads = heads, 
    dim_key = dim_key, dim_value = dim_value, dropout = dropout
)

model_dict = torch.load('../models/coco_image_transformer_best_loss_state_dict.pth')

model.load_state_dict(model_dict['model_dict'])

model = model.to('cpu')
model = model.eval()


def getImageTensor(imgPath, tfms, device):

    return tfms(Image.open(imgPath).convert("RGBA")).unsqueeze(0).to(device)


def predictTags(imagePath, tfms, device = 'cpu', maxLen = 20):

    imgTnsr = getImageTensor(imgPath, tfms, 'cpu')
    print("done")
    trg_ixs = [tokenizer.token_to_id('[CLS]')]

    for i in range(maxLen):

        trg_seq = torch.FloatTensor(trg_ixs).unsqueeze(0).to(device)

        with torch.no_grad():
            op = model(imgTnsr, trg_seq)

        # print(f'Iteration {i}: {op.max(1)[-1]}')
        pred = op.max(1)[-1][-1].item()

        trg_ixs.append(pred)

        if pred == tokenizer.id_to_token('[SEP]'):

            break

    return tokenizer.decode(trg_ixs)

imgPath = '../data/images/1369.png'

img = Image.open(imgPath).convert("RGBA")

print(predictTags(imgPath, simple_transforms))