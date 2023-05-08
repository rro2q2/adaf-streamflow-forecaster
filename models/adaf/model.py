import torch
import torch.nn as nn
import numpy as np
from typing import Dict
import torch.nn.functional as F

from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.utils.config import Config


class Encoder(nn.Module):
    """ Encode the time series input sequence. """

    def __init__(self, cfg: Config):
        super(Encoder, self).__init__()
        self.cfg = cfg
        self.embedding_net = InputLayer(self.cfg)
        self.lstm_encoder = nn.LSTM(input_size = self.embedding_net.output_size, hidden_size = self.cfg.hidden_size, batch_first=True)

    def forward(self, data: Dict[str, torch.Tensor], hidden) -> Dict[str, torch.Tensor]:
        # data.shape = [batch_size, seq_len, hidden_dim]
        # hidden.shape = [batch_size, 1, hidden_dim]
        output, (h_n, c_n) = self.lstm_encoder(data, hidden) # output = [seq_len, batch_size, hidden_dim]; (h_n, c_n)
        pred = {'output': output, 'h_n': h_n, 'c_n': c_n}
        return pred
    
    def init_hidden(self, data, n_layers=1):
        if data.shape[0] != self.cfg.batch_size:
            return (
                torch.zeros(n_layers, data.shape[0], self.cfg.hidden_size, device=self.cfg.device),
                torch.zeros(n_layers, data.shape[0], self.cfg.hidden_size, device=self.cfg.device)
            )
        return (
            torch.zeros(n_layers, self.cfg.batch_size, self.cfg.hidden_size, device=self.cfg.device),
            torch.zeros(n_layers, self.cfg.batch_size, self.cfg.hidden_size, device=self.cfg.device)
        )
    
class Attention(nn.Module):
    """ Applies additive attention to decoder from encoder.
        Source for additive attention:
        https://arxiv.org/pdf/1409.0473.pdf?ref=floydhub-blog.
    """
    def __init__(self, cfg: Config):
        super(Attention, self).__init__()
        self.cfg = cfg
        self.fc_hidden = nn.Linear(self.cfg.hidden_size, self.cfg.hidden_size, bias=False) # query
        self.fc_encoder = nn.Linear(self.cfg.hidden_size, self.cfg.hidden_size, bias=False) # key, values
        self.weight = nn.Parameter(torch.FloatTensor(1, self.cfg.hidden_size))
        self.fc_combined = nn.Linear(self.cfg.hidden_size, 1)
        
        self.attn = nn.Linear(self.cfg.hidden_size * 2, self.cfg.hidden_size)
        self.v = nn.Linear(self.cfg.hidden_size, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        # hidden.shape = (num_layers, batch_size, hidden_dim)
        # encoder_outputs.shape = (batch_size, seq_len, hidden_dim)
        seq_len = encoder_outputs.size(1) 
        h = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        # energy.shape = (batch_size, seq_len, hidden_dim)
        energy = torch.tanh(self.attn(torch.cat((h, encoder_outputs), dim=2)))
        # attention_weights.shape = (batch_size, seq_len, 1)
        attention_weights = F.softmax(self.v(energy), dim=1)
        # context_vector.shape = (batch_size, hidden_dim)
        context_vector = (attention_weights * encoder_outputs).sum(dim=1)
        return context_vector

class Decoder(nn.Module):
    """ Decodes hidden state from encoder and attention module """
    def __init__(self, cfg: Config):
        super(Decoder, self).__init__()
        self.cfg = cfg
        self.embedding_net = InputLayer(self.cfg)
        # self.decoder = decoder
        self.lstm_decoder = nn.LSTM(input_size=self.embedding_net.output_size + self.cfg.hidden_size, hidden_size=self.cfg.hidden_size, batch_first=True)
        self.linear = nn.Linear(self.cfg.hidden_size, self.embedding_net.output_size)
        self.attention = Attention(cfg=cfg)

        self.dropout = nn.Dropout(p=cfg.output_dropout)
        self.output_size = len(self.cfg.target_variables)
        self.head = get_head(cfg=cfg, n_in=self.cfg.hidden_size, n_out=self.output_size)
              

    def forward(self, data: Dict[str, torch.Tensor], hidden: torch.Tensor, encoder_outputs: torch.Tensor):
        context = self.attention(hidden[0], encoder_outputs)
        data = data.unsqueeze(1)
        # Concatenate the input data and the context vector
        # context.shape = (batch_size, 1, hidden_dim)
        data = torch.cat((data, context.unsqueeze(1)), -1)
        
        # [seq_len, batch_size, hidden_len]
        output, (h_n, c_n) = self.lstm_decoder(data, hidden)
       
        # Reshape to [batch_size, seq, n_hiddens]
        h_n = h_n.transpose(0, 1) # [batch_size, 1, hid_size]
        c_n = c_n.transpose(0, 1) # [batch_size, 1, hid_size]
        
        pred = {'output': output, 'h_n': h_n, 'c_n': c_n}
        pred.update(self.head(self.dropout(output)))
        decoder_output = self.linear(output.squeeze(0)).transpose(0, 1)

        return pred, decoder_output, context.unsqueeze(1)
    
    
class Discriminator(nn.Module):
    """ Classify between source and target """
    def __init__(self, cfg: Config):
        super(Discriminator, self).__init__()
        self.cfg = cfg
        self.embedding_net = InputLayer(self.cfg)
        self.classes = 1
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=self.cfg.hidden_size, out_features=self.cfg.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.cfg.output_dropout),
            nn.Linear(in_features=self.cfg.hidden_size, out_features=self.classes),
            nn.Sigmoid()
        )
        
    def forward(self, data):
        output = self.discriminator(data)
        return output.squeeze(-1)
    
    
class ADAF(nn.Module):
    def __init__(self, cfg: Config):
        super(ADAF, self).__init__()
        # self.dict_config = dict_config
        self.cfg = cfg
        self.iw = 335 # input window
        self.ow = 30  # output window
        self.device = self.cfg.device #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tf_ratio = 0.5
        
        self.encoder = Encoder(cfg=self.cfg).to(self.device)
        self.decoder = Decoder(cfg=self.cfg).to(self.device)
        self.discriminator = Discriminator(cfg=self.cfg).to(self.device)
        
    def forward(self, data):
        if data['x_d'].shape[2] != self.encoder.embedding_net.output_size:
            data['x_d'] = self.encoder.embedding_net(data)
        
        seq_pred = dict()
        # input_seq, target_seq -> [batch_size, seq_len, num_features]
        input_seq = data['x_d'][:self.iw, :, :].transpose(0, 1)
        target_seq = data['x_d'][self.iw:, :, :].transpose(0, 1)
       
        # Initialize encoder hidden state
        init_hidden = self.encoder.init_hidden(input_seq, n_layers=1)
        # Encode input with init hidden state
        encoder_pred = self.encoder(input_seq, init_hidden)
    
        encoder_outputs = encoder_pred['output'] # [seq_len, batch_size, hidden_dim]
        encoder_hidden = (encoder_pred['h_n'], encoder_pred['c_n'])
        # Assign last hidden state of encoder into the start of the decoder hidden state
        decoder_hidden = encoder_hidden
        # Last input to feed into the decoder
        decoder_input = input_seq[:, -1, :] # [batch_size, num_dim]

        # predict target sequence
        for t in range(self.ow): 
            decoder_pred, decoder_output, context = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_hidden = (decoder_pred['h_n'].transpose(0, 1), decoder_pred['c_n'].transpose(0, 1))
            # Update the sequence generator prediction map
            if 'y_hat' not in seq_pred:
                seq_pred['y_hat'] = decoder_pred['y_hat']
            else:
                seq_pred['y_hat'] = torch.cat((seq_pred['y_hat'], decoder_pred['y_hat']), dim=1)
            seq_pred['output'] = decoder_pred['output'] # [32, 1, 128]
            seq_pred['h_n'] = decoder_pred['h_n']
            seq_pred['c_n'] = decoder_pred['c_n']
            if 'decoder_output' not in seq_pred:
                seq_pred['decoder_output'] = decoder_output
            else:
                seq_pred['decoder_output'] = torch.cat((seq_pred['decoder_output'], decoder_output), dim=0)
            if 'attn_data' not in seq_pred:
                seq_pred['attn_data'] = context
            else:
                seq_pred['attn_data'] = torch.cat((seq_pred['attn_data'], context), dim=1)
            
            # Update decoder input
            decoder_input = target_seq[:, t, :]
            
        # Flatten tensor to combine batch size with sequence length
        seq_pred['attn_data'] = seq_pred['attn_data'].flatten(0, 1)
        return seq_pred
    