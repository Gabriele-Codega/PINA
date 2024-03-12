import torch.nn as nn
import torch
import math

## BUG: the Flatten layer in the Encoder only works with batched data.
# Reason is that it tries to flatten starting from dim=1, which only makes
# sense if the data is batched (i.e. 3dimensional).
# Workaround: pass a single input as a slice, i.e cae(u[:1,:,:]) would ideally be
# cae(u[0,:,:]).


class CAE(nn.Module):
    def __init__(self,
                full_dimension,
                latent_dimension,
                encoder_kwargs,
                decoder_kwargs):
        """Construct Convolutional AutoEncoder

        :param full_dimension: dimension of the origin space
        :param latent_dimension: dimension of the latent space
        :param encoder_kwargs: list of dictionaries with parameters for convolutional layers in the encoder.
        :param decoder_kwargs: list of dictionaries with parameters for convolutional layers in the decoder.

        .. note:: 
            Each layer requires one dictionary (which is indeed annoying and will possibly be modified).
            The user is responsible for the tuning of parameters so that the output dimension of the decoder
            is the same as the input dimension of the encoder.
        """
        
        super().__init__()

        self.full_dimension = full_dimension
        self.latent_dimension = latent_dimension

        self._encoder = Encoder(full_dimension, latent_dimension, *encoder_kwargs)
        self._decoder = Decoder(full_dimension, self._encoder._after_conv_size, latent_dimension, *decoder_kwargs)

    def forward(self,x):
        # Check to see if in the case where we have u(t),u(t+1).
        # Only works for batched input, because if the input is not batched
        # the size would still be 3 but the first dimension is treated as batch size
        # and the whole thing breaks
        
        shape = list(x.size()) if isinstance(x,torch.Tensor) else list(x.shape)
        if (len(shape) == 4):
            n = shape[1]
            output_ = []
            for i in range(n):
                #print(x.tensor[:,i,...])
                y = self.encoder(x.tensor[:,i,...])
                y = self.decoder(y)
                output_.append(y)
            output = torch.stack(output_)
        else:
            x = self.encoder(x)
            output = self.decoder(x)
        return output
    
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
