import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

SEED = 3007
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

## image positional embedding sine
class PositionEmbeddingSine(nn.Module):

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list):
        x = tensor_list
        b, c, h, w = x.size()
        mask = torch.zeros((b, h, w), dtype=torch.bool, device=x.device)
        # assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


## positional encoding
class PositionalEncoding(nn.Module):
    '''
    Positional encoding table for attention, (not trained)
    '''

    def __init__(self, dim_hid, num_pos):

        super(PositionalEncoding, self).__init__()

        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(num_pos, dim_hid))

    def _get_sinusoid_encoding_table(self, num_pos, dim_hid):

        def get_position_angle_vec(position):

            return [
                position / np.power(10000, 2 * (hid_j // 2) / dim_hid) for hid_j in range(dim_hid)
            ]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_pos)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)


    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


## scalded dot product attention

class ScaledDotProductAttention(nn.Module):

    ''' Scaled Attention ''' 

    def __init__(self, temperature, attention_dropout = 0.1):

        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, query, key, value, mask = None):

        attention = torch.matmul(query / self.temperature, key.transpose(2, 3))

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)

        attention = self.dropout(torch.softmax(attention, dim = -1))

        output = torch.matmul(attention, value)

        return output, attention



## position wise feedforward
class PositionWiseFeedForward(nn.Module):

    '''
    Feed Forward Layer
    '''

    def __init__(self, dim_in, dim_hid, elu_func: str = 'gelu', dropout = 0.1):

        super().__init__()

        self.feedforward_1 = nn.Linear(dim_in, dim_hid)
        self.feedforward_2 = nn.Linear(dim_hid, dim_in)

        self.layer_norm = nn.LayerNorm(dim_in, eps = 1e-6)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ReLU()


    def forward(self, inp):

        residual = inp
        inp = self.layer_norm(inp)

        inp = self.feedforward_2(self.elu(self.feedforward_1(inp)))

        inp = self.dropout(inp)

        inp += residual

        return inp
    


## multi-head attention
class MultiHeadAttention(nn.Module):

    ''' Multi-Head Attention '''

    def __init__(self, heads, dim_model, dim_key, dim_value, dropout = 0.1):

        super().__init__()

        self.heads = heads
        self.dim_key = dim_key
        self.dim_value = dim_value

        self.toquery = nn.Linear(dim_model, heads * dim_key, bias = False)
        self.tokey = nn.Linear(dim_model, heads * dim_key, bias = False)
        self.tovalue = nn.Linear(dim_model, heads * dim_value, bias = False)


        self.union = nn.Linear(heads * dim_value, dim_model, bias = False)

        self.attention = ScaledDotProductAttention(temperature = dim_key ** 0.5)

        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(dim_model, eps = 1e-6)

    def forward(self, query, key, value, mask=None):

        dim_key, dim_value, heads = self.dim_key, self.dim_value, self.heads

        batch_size, length_query, length_key, length_value = query.size(0), query.size(1), key.size(1), value.size(1)

        residual = query

        query = self.layer_norm(query)

        query = self.toquery(query).view(batch_size, length_query, heads, dim_key)
        key = self.tokey(key).view(batch_size, length_key, heads, dim_key)
        value = self.tovalue(value).view(batch_size, length_value, heads, dim_value)

        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        query, attention = self.attention(query, key, value, mask = mask)

        query = query.transpose(1, 2).contiguous().view(batch_size, length_query, -1)
        query = self.dropout(self.union(query))

        query += residual

        return query, attention


## encoder layer
class EncoderLayer(nn.Module):

    ''' Two Layer Architecture '''

    def __init__(self, dim_model, dim_inner, heads, dim_key, dim_value, dropout = 0.1):

        super(EncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(heads, dim_model, dim_key, dim_value, dropout = dropout)
        self.pos_ff = PositionWiseFeedForward(dim_model, dim_inner, dropout = dropout)

    def forward(self, encoder_input, self_attention_mask = None):

        encoder_output, encoder_self_attention = self.self_attention(
            encoder_input, encoder_input, encoder_input, mask = self_attention_mask
        )

        encoder_output = self.pos_ff(encoder_output)

        return encoder_output, encoder_self_attention


## Decoder Layer
class DecoderLayer(nn.Module):

    def __init__(self, dim_model, dim_inner, heads, dim_key, dim_value, dropout = 0.1):

        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(heads, dim_model, dim_key, dim_value, dropout = dropout)
        self.encoder_attention = MultiHeadAttention(heads, dim_model, dim_key, dim_value, dropout = dropout)

        self.pos_ff = PositionWiseFeedForward(dim_model, dim_inner, dropout = dropout)

    def forward(
        self, decoder_input, encoder_output, 
        self_attention_mask = None, decoder_encoder_attention_mask = None
        ):

        decoder_output, decoder_self_attention = self.self_attention(
            decoder_input, decoder_input, decoder_input, mask = self_attention_mask
        )

        decoder_output, decoder_encoder_attention = self.encoder_attention(
            decoder_output, encoder_output, encoder_output, mask = decoder_encoder_attention_mask
        )

        decoder_output = self.pos_ff(decoder_output)

        return decoder_output, decoder_self_attention, decoder_encoder_attention