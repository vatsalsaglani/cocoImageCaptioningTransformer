import torch
import torch.nn as nn
import torch.nn.functional as F
from ImageEncoder import ImageEncoder
from Decoder import Decoder
from CustResnetModel import ResNetCust2

SEED = 3007
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


class Transformer(nn.Module):

    def __init__(
        self, target_vocab_size, target_pad_id, emb_dim, 
        hid_dim, dim_model, dim_inner, layers, heads,
        dim_key, dim_value, dropout = 0.15, 
        num_pos = 100, target_emb_projection_weight_sharing = False):



        super(Transformer, self).__init__()

        self.backbone = ResNetCust2(emb_dim)
        
        self.embed = nn.Conv2d(2048, emb_dim, 1)
        

        self.target_pad_id = target_pad_id
        self.encoder = ImageEncoder(
            emb_dim = emb_dim, hid_dim = hid_dim, layers = layers, heads = heads, dim_key = dim_key,
            dim_value = dim_value, dim_model = dim_model, dim_inner = dim_inner, dropout = dropout
        )

        self.decoder = Decoder(
                target_vocab_size, emb_dim, layers, heads, dim_key, dim_value, dim_model, dim_inner, target_pad_id, dropout = dropout, num_pos = num_pos
        )

        self.target_word_projection = nn.Linear(dim_model, target_vocab_size, bias = False)

        # for parameter in self.parameters():

        #     if parameter.dim() > 1:

        #         nn.init.xavier_uniform_(parameter)

        assert dim_model == emb_dim, f'Dimension of all the module outputs mush be same'

        self.x_logit_scale = 1

        if target_emb_projection_weight_sharing:

            self.target_word_projection.weight = self.decoder.word_embedding.weight

            self.x_logit_scale = (dim_model ** -0.5)


    def get_pad_mask(self, sequence, pad_id):

        return (sequence != pad_id).unsqueeze(-2)

    def get_subsequent_mask(self, sequence):

        batch_size, seq_length = sequence.size()

        subsequent_mask = (
            1 - torch.triu(
                torch.ones((1, seq_length, seq_length), device = sequence.device), diagonal = 1
            )
        ).bool()

        return subsequent_mask

    def forward(self, source_seq, target_seq):

        
        imageFeatures = self.backbone(source_seq)
        features = self.embed(imageFeatures)


        
        target_mask = self.get_pad_mask(target_seq, self.target_pad_id) & self.get_subsequent_mask(target_seq)

        
        encoder_output = self.encoder(features, None)
        decoder_output = self.decoder(target_seq, target_mask, encoder_output, None)

        seq_logit = self.target_word_projection(decoder_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))