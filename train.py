import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math 
import os 
import re 
import pandas as pd 
import numpy as np 
from PIL import Image 
from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm, trange 
from utilities import patch_trg_n, count_model_parameters, patch_src, device, cal_performance, cal_loss
from Transformer import Transformer 
from ImageCaptionDataset import ImageCaptionDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import logging
import torch.optim as optim
from torch.autograd import Variable

logging.basicConfig(filename='initial_image_caption_transformer.log', filemode='a', level=logging.INFO, format = '{asctime} {filename} {message}', style="{")


SEED = 3007
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

data = pd.read_csv('../data/cocoImageCaptions40k.csv')
data.reset_index(inplace = True)
if 'index' in data.columns:
    data.drop(['index'], axis = 1, inplace = True)

image_folder = '../data/images'

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
bs = 64
n_wk = 5

model = Transformer(
    target_vocab_size=target_vocab_size, target_pad_id=target_pad_id,
    emb_dim=emb_dim, hid_dim=hid_dim, dim_model=dim_model,
    dim_inner=dim_inner, layers = layers, heads = heads, dim_key=dim_key,
    dim_value = dim_value, dropout = dropout
)

model_dict = torch.load('../models/coco_image_transformer_20k_best_loss_state_dict.pth')

model.load_state_dict(model_dict['model_dict'])

model = model.to(device())
model = model.train()

print(f'Total Model Parameters: {count_model_parameters(model) / 1e6} Million')
logging.info(f'Total Model Parameters: {count_model_parameters(model) / 1e6} Million')


train_ds = ImageCaptionDataset(image_folder, data, tokenizer, simple_transforms, seq_len)
train_dl = DataLoader(train_ds, batch_size = bs, num_workers = n_wk, pin_memory = True, shuffle = True)


optimizer = optim.Adam(model.parameters(), lr = 0.0005, betas = (0.88, 0.98), eps = 1e-9)


## training
clip = 1.0
EPOCHS = 50
EPOCHS_START = 0
loss_, acc_, perp_ = [], [], []
loss_.append(model_dict['loss'])

for epoch in trange(EPOCHS_START, EPOCHS + EPOCHS_START, desc = "Epoch", leave = False):

    total_loss, n_label_total, n_label_correct, n_perplexity = 0, 0, 0, 0
    model = model.train()

    for ix, batch in enumerate(tqdm(train_dl, desc = 'Train DL', leave = False)):

        src_seq = patch_src(batch['src'].to(device()))
        trg_seq, gold = map(lambda x: x.to(device()), patch_trg_n(batch['trg'], seq_len=seq_len))

        src_seq, trg_seq, gold = Variable(src_seq), Variable(trg_seq), Variable(gold)

        optimizer.zero_grad()

        pred = model(src_seq, trg_seq)

        loss, n_correct, n_label = cal_performance(pred, gold.long(), target_pad_id)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        n_label_total += n_label
        n_label_correct += n_correct
        # n_perplexity += np.exp(loss.item())

        total_loss += loss.item()

    loss_per_label = total_loss / n_label_total
    accuracy = n_label_correct / n_label_total
    avgPerplexity = np.exp(loss_per_label)

    text = f'Training EPOCH: {epoch} | Loss: {loss_per_label} | Perplexity: {avgPerplexity}'
    logging.info(text)
    print(text)

    if epoch == 0:
        torch.save(model, '../models/coco_image_transformer_initial_loss.pt')
        torch.save({
            "model_dict": model.state_dict(),
            "optimizer_dict": optimizer.state_dict(),
            "loss": loss_per_label,
            "epoch": epoch,
            "perplexity": avgPerplexity
        }, '../models/coco_image_transformer_initial_loss_state_dict.pth')

    if len(loss_) > 0 and loss_per_label < min(loss_):

        torch.save(model, '../models/coco_image_transformer_best_loss.pt')
        torch.save({
            "model_dict": model.state_dict(),
            "optimizer_dict": optimizer.state_dict(),
            "loss": loss_per_label,
            "epoch": epoch,
            "perplexity": avgPerplexity
        }, '../models/coco_image_transformer_best_loss_state_dict.pth')

    loss_.append(loss_per_label)