import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from CustResnetModel import ResNetCust2
from AttentionModules import EncoderLayer, ImagePositionEmbeddingLearned, PositionEmbeddingSine

SEED = 3007
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

class ImageEncoder(nn.Module):

    def __init__(self, emb_dim, hid_dim, layers, heads, dim_key, dim_value, dim_model, dim_inner, dropout = 0.1):

        super(ImageEncoder, self).__init__()


        self.position_encoding = PositionEmbeddingSine(emb_dim//2)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(
                dim_model, dim_inner, heads, dim_key, dim_value, dropout = dropout
            ) for _ in range(layers)
        ])

        self.layer_norm = nn.LayerNorm(dim_model, eps = 1e-6)

    def forward(self, features, source_mask = None, return_attentions = False):

        encoder_self_attention_list = []


        pos_op = self.position_encoding(features)

        encoder_output = features.flatten(2).permute(2, 0, 1) + pos_op.flatten(2).permute(2, 0, 1)

        encoder_output = encoder_output.permute(1, 0, 2)

        for encoder_layer in self.layer_stack:
            
            encoder_output, encoder_self_attention = encoder_layer(encoder_output, self_attention_mask=source_mask)
            encoder_self_attention_list += [encoder_self_attention] if return_attentions else []
            
        encoder_output = self.layer_norm(encoder_output)
        
        if return_attentions:
            
            return encoder_output, encoder_self_attention_list
        
        return encoder_output