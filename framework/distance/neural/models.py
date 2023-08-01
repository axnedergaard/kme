import math

import torch
import torch.nn as nn
from torch import Tensor

import torch.nn.functional as F
from torchkme.encoder import activations
from torchkme.encoder import layers
from torchkme.encoder import utils

# .TODO initialize weights according to Xavier


# --- Activation functions ---

_supported_activations = {
    'relu': F.relu,
    'tanh': F.tanh,
    'sigmoid': F.sigmoid,
    'softplus': F.softplus,
    'leaky_relu': F.leaky_relu,
    'elu': F.elu
}


def get_phi(activation: str) -> nn.Module:
    # Get the activation function from the supported ones
    return _supported_activations[activation]


# --- Base FeedForward Network ---

class FeedForward(nn.Module):

    def __init__(self, **kwargs) -> None:        
        super(FeedForward, self).__init__()
        self._verify_parameters(kwargs)

        # the input and output dimensions
        self.input_dim = kwargs['input_dim']
        self.output_dim = kwargs['output_dim']

        # number layers with their dimensions
        self.num_layers = kwargs['n_hid_layers']
        self.hidden_dims = kwargs['hidden_dims']

        # activation function (same for all)
        self.phi = get_phi(kwargs['activation'])

        # network layers (input, hidden, output)
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dims[0])
        self.output_layer = nn.Linear(self.hidden_dims[-1], self.output_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]) 
            for i in range(self.num_layers - 1)
        ])


    def forward(self, x: Tensor) -> Tensor:
        # Forward pass through the network
        z = self.phi(self.input_layer(x))
        for layer in self.hidden_layers:
            z = self.phi(layer(z))
        return self.output_layer(z)
    

    def _verify_parameters(self, kwargs) -> None:
        # Verify the parameters passed to the FeedForward constructor
        assert kwargs.get("input_dim") > 0, "Input dimension must be positive."
        assert kwargs.get("output_dim") > 0, "Output dimension must be positive."
        assert kwargs.get("n_hid_layers") > 0, "Number of hidden layers must be positive."
        assert len(kwargs.get("hidden_dims")) == kwargs.get("n_hid_layers"), "Must specify n_hid_layers dimensions."
        assert kwargs.get("activation") in _supported_activations.keys(), "Activation function not supported."


# --- Vanilla AutoEncoder ---

class AutoEncoder(nn.Module):

    def __init__(self, **kwargs) -> None:
        super(AutoEncoder).__init__(**kwargs)
        self._verify_parameters(kwargs)

        # encoder initialization
        self.encoder = FeedForward({
            "input_dim": kwargs['input_dim'],
            "output_dim": kwargs['latent_dim'],
            "n_hid_layers": kwargs['n_hid_layers'],
            "hidden_dims": kwargs['hidden_dims'],
            "activation": kwargs['activation']
        })
        
        # decoder initialization
        self.decoder = FeedForward({
            "input_dim": kwargs['latent_dim'],
            "output_dim": kwargs['input_dim'],
            "n_hid_layers": kwargs['n_hid_layers'],
            "hidden_dims": kwargs['hidden_dims'][::-1],
            "activation": kwargs['activation']
        })


    def forward(self, x: Tensor) -> Tensor:
        # Forward pass through the network
        return self.decoder(self.encoder(x))
    

    def encode(self, x: Tensor) -> Tensor:
        # Forward pass through the encoder only
        return self.encoder(x)
    

    def decode(self, z: Tensor) -> Tensor:
        # Forward pass through the decoder only
        return self.decoder(z)
    

    def _verify_parameters(self, kwargs) -> None:
        # Verify the parameters passed to the AutoEncoder constructor
        assert kwargs.get("input_dim") > 0, "Input dimension must be positive."
        assert kwargs.get("latent_dim") > 0, "Latent dimension must be positive."


# --- Variational AutoEncoder ---

class VAE(nn.module):

    def __init__(self, **kwargs) -> None:
        self(VAE).__init__(**kwargs)
        self._verify_parameters(kwargs)
        raise NotImplementedError()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()
    
    def encode(self, x: Tensor) -> Tensor:
        raise NotImplementedError()
    
    def decode(self, z: Tensor) -> Tensor:
        raise NotImplementedError()
    
    def _verify_parameters(self, kwargs) -> None:
        raise NotImplementedError()
    

# --- Denoising AutoEncoder ---

class DAE(nn.Module):

    def __init__(self, **kwargs) -> None:
        self(DAE).__init__(**kwargs)
        self._verify_parameters(kwargs)
        raise NotImplementedError()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()
    
    def encode(self, x: Tensor) -> Tensor:
        raise NotImplementedError()
    
    def decode(self, z: Tensor) -> Tensor:
        raise NotImplementedError()
    
    def _verify_parameters(self, kwargs) -> None:
        raise NotImplementedError()
    

# --- Contractive AutoEncoder ---

class CAE(nn.Module):

    def __init__(self, **kwargs) -> None:
        self(CAE).__init__(**kwargs)
        self._verify_parameters(kwargs)
        raise NotImplementedError()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()
    
    def encode(self, x: Tensor) -> Tensor:
        raise NotImplementedError()
    
    def decode(self, z: Tensor) -> Tensor:
        raise NotImplementedError()
    
    def _verify_parameters(self, kwargs) -> None:
        raise NotImplementedError()
    

# --- Compact Transformer ---

class PositionalEncoding(nn.Module):

    def __init__(self, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MiniTransformerM(nn.Module):

    def __init__(self, **kwargs) -> None:
        super(MiniTransformerM).__init__(**kwargs)
        self._verify_parameters(kwargs)

        self.pos_enc = PositionalEncoding(kwargs['input_dim'])
        self.attention = nn.MultiheadAttention(kwargs['input_dim'], kwargs['n_heads'])
        self.norm1 = nn.LayerNorm(kwargs['input_dim'])

        self.ffn = nn.Sequential(
            nn.Linear(kwargs['input_dim'], kwargs['hidden_dims'][0]),
            nn.ReLU(),
            nn.Linear(kwargs['hidden_dims'][0], kwargs['input_dim'])
        )

        self.norm2 = nn.LayerNorm(kwargs['input_dim'])

    
    def forward(self, x: Tensor) -> Tensor:
        x = self.pos_enc(x)
        attention_o, _ = self.attention(x, x, x) # self-attention
        x = self.norm1(x + attention_o)
        ffn_o = self.ffn(x) # feed-forward network
        return self.norm2(x + ffn_o)
    

class MiniTransformer(nn.Module):
    def __init__(self, dim, latent_dim, heads=8):
        super().__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        self.pos_enc = PositionalEncoding(dim)
        self.transformer = nn.Transformer(
            d_model=dim, 
            nhead=heads, 
            num_encoder_layers=1, 
            num_decoder_layers=1
        )
        self.encoder_linear = nn.Linear(dim, latent_dim)
        self.decoder_linear = nn.Linear(latent_dim, dim)
        self.norm = nn.LayerNorm(normalized_shape=dim)

    def forward(self, x):
        # Assuming x is of shape [sequence_len, batch_size, dim]
        x = self.pos_enc(x)
        output = self.transformer(x, x)  # Self-attention
        return self.norm(output)

    def encode(self, x):
        # Pass the input through the positional encoding and the transformer encoder
        x = self.pos_enc(x)
        encoded = self.transformer.encoder(x)
        # Apply linear transformation
        return self.encoder_linear(encoded)

    def decode(self, x):
        # Apply inverse linear transformation
        x = self.decoder_linear(x)
        # Pass the input through the transformer decoder and the layer normalization
        return self.norm(self.transformer.decoder(x))
