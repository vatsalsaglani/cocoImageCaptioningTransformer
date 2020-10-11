import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import re


def pad_trg_seq(tnsr, seq_len = 100):

    if tnsr.size()[0] > seq_len:
        
        tnsr = tnsr[:seq_len]
        
        return tnsr

#     arr = torch.tensor(arr)

    op = torch.zeros((seq_len))

    op[:tnsr.size(0)] = tnsr

    return op.float()

def patch_trg_n(tnsr, seq_len, eos_id = 3):
    
    where = torch.where(tnsr == eos_id)
    
    trg = torch.stack([pad_trg_seq(tnsr[where[0][i]][:where[1][i]], seq_len=seq_len) for i in range(len(where[0]))])
    gold = torch.stack([pad_trg_seq(tnsr[where[0][i]][1:], seq_len = seq_len) for i in range(len(where[0]))])
    
    return trg, gold.contiguous().view(-1)

def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word

def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss

def patch_src(src):
    """Patching source with pad if needed"""

    return src

def device():
    '''
    keeping code device agnostic 
    '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def count_model_parameters(model):

    ''' Count of parameters which can be trained in the model defination '''

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
