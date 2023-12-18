import torch.nn as nn
import math

class CAE(nn.Module):
    def __init__(self,
                full_dimension,
                latent_dimension,
                encoder_kwargs,
                decoder_kwargs):
        
        super().__init__()

        self.full_dimension = full_dimension
        self.latent_dimension = latent_dimension

        self._encoder = Encoder(full_dimension, latent_dimension, *encoder_kwargs)
        self._decoder = Decoder(full_dimension, self._encoder._after_conv_size, latent_dimension, *decoder_kwargs)

    def forward(self,x):
        x = self.encoder(x)
        return self.decoder(x)
    
    @property
    def encoder(self):
        return self._encoder
    
    @property
    def decoder(self):
        return self._decoder
    

class Encoder(nn.Module):
    def __init__(self,
                full_dimension,
                latent_dimension,
                *conv_kwargs):
        super().__init__()

        layers = []
        current_size = full_dimension
        for args in conv_kwargs:
            layers.append(nn.Conv1d(**args))

            padding = args['padding']
            dilation = args['dilation']
            kernel_size = args['kernel_size']
            stride = args['stride']

            current_size = math.floor((current_size + 2*padding - dilation * (kernel_size - 1)-1)/stride + 1)
        self._after_conv_size = current_size
        layers.append(nn.Flatten())
        layers.append(nn.Linear(current_size*conv_kwargs[-1]['out_channels'],latent_dimension))

        layers_functions = []
        for layer in layers:
            layers_functions.append(layer)
            layers_functions.append(nn.ELU())

        self.cnn = nn.Sequential(*layers_functions)

    def forward(self,x):
        return self.cnn(x)

class Decoder(nn.Module):
    def __init__(self,
                full_dimension,
                intermediate_dim,
                latent_dimension,
                *conv_kwargs):
        super().__init__()

        layers = []
        layers.append(nn.Linear(latent_dimension, conv_kwargs[0]['in_channels'] * intermediate_dim))
        layers.append(nn.Unflatten(1,(conv_kwargs[0]['in_channels'], intermediate_dim)))

        current_size = intermediate_dim
        for args in conv_kwargs:
            layers.append(nn.ConvTranspose1d(**args))

            padding = args['padding']
            dilation = args['dilation']
            kernel_size = args['kernel_size']
            stride = args['stride']
            outpadding = args['output_padding']

            current_size = (current_size-1)*stride - 2*padding + dilation*(kernel_size-1) + outpadding +1

        layers_functions = []
        for layer in layers[:-1]:
            layers_functions.append(layer)
            layers_functions.append(nn.ELU())
        layers_functions.append(layers[-1])

        self.cnn = nn.Sequential(*layers_functions)



    def forward(self,x):
        return self.cnn(x)