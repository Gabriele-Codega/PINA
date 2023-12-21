"""
   Testing the convolutional autoencoder structure 
   in the PINA framework (v. 0.1).
   The model is a Burgers equation in 1D.

"""

from pina.model import CAE
from pina.problem import SpatialProblem,TimeDependentProblem
from pina.solvers import PINN
from pina.condition import Condition
from pina.geometry import CartesianDomain
from pina import LabelTensor
from pina import Trainer
from pina.plotter import Plotter
from pina.callbacks import MetricTracker
from pina.equation import Equation
from torchinfo import summary
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import torch

## load the data
data = loadmat("Burgers_FOM.mat")
u = data['u_Mat'].T # (nt,nx)
nt = u.shape[0] 
nx = u.shape[1]
L = data['L'] # (nx,1)
L = L.reshape(nx)
t = data['time'] # (nt,1)
t = t.reshape(nt)

## getting domain boundaries to define the domain in the PINA problem
x_dom = [L[0], L[-1]]
t_dom = [t[0], t[-1]]

## making L and t LabelTensors as required by PINA
L = LabelTensor(L,'x')
t = LabelTensor(t,'t')

## rehaping u to have 1 channel, as required by Conv1D
## also making u a LabelTensor as reuqired by PINA
u = u.reshape((nt,1,nx))
labels = [f'u{i}' for i in range(u.shape[-1])]
u = LabelTensor(u,labels)


## lists of dictionaries with parameters to build autoencoder layers.
## TODO: implement a less tedious way to handle the construction of layers
# encoder parameters
en_args = [{'in_channels':1,'out_channels':4,'kernel_size':3,'stride':1,'padding':0,'dilation':1},
           {'in_channels':4,'out_channels':8,'kernel_size':3,'stride':1,'padding':1,'dilation':1},
           {'in_channels':8,'out_channels':16,'kernel_size':3,'stride':1,'padding':1,'dilation':1},
           {'in_channels':16,'out_channels':32,'kernel_size':3,'stride':1,'padding':1,'dilation':1}]
# decoder parameters
dec_args = [{'in_channels':32,'out_channels':16,'kernel_size':3,'stride':1,'padding':1,'dilation':1,'output_padding':0},
           {'in_channels':16,'out_channels':8,'kernel_size':3,'stride':1,'padding':1,'dilation':1,'output_padding':0},
           {'in_channels':8,'out_channels':4,'kernel_size':3,'stride':1,'padding':1,'dilation':1,'output_padding':0},
           {'in_channels':4,'out_channels':1,'kernel_size':3,'stride':1,'padding':0,'dilation':1,'output_padding':0}]

cae = CAE(30,8,en_args,dec_args)

## defining the (Physics Informed) Model Order Reduction problem
class PIMOR(SpatialProblem,TimeDependentProblem):
    # domain definition is required as they are abstractmethods
    # but they are not used currently.
    # Possible solution is to inherit from AbstractProblem
    spatial_domain = CartesianDomain({'x': x_dom})
    temporal_domain = CartesianDomain({'t': t_dom})
    input_variables = labels
    output_variables = labels

    # dummy, meaningless equation to test the handling of 'physics based' loss.
    def dummy_eq(input_,output_):
        usq = torch.sum(input_ * input_,dim=-1)
        z = cae(input_)
        zsq = torch.sum(z*z,dim=-1)
        return usq - zsq

    # We can pass points from the spatial/temporal/spatio-temporal domain
    # as LabelTensors with something like `Condition(input_points=pts, equation=Burgers)`.
    # This way we do not actually need to sample the domains and we do not necessarily 
    # have to work with the pairs (x,t) which would come from PINA internals.
    conditions = {'data': Condition(input_points=u,output_points = u,data_weight=0.8),
                  'dum': Condition(input_points=u,equation=Equation(dummy_eq),data_weight=0.2)}

rom = PIMOR()


solver = PINN(problem=rom,
              model=cae,
              optimizer=torch.optim.Adam,
              optimizer_kwargs={'lr':0.001},
              scheduler_kwargs={'factor':0.5})
trainer = Trainer(solver,batch_size=20,max_epochs=100,callbacks = [MetricTracker()])


trainer.train()
torch.save(cae, 'pinn_cae_pyhs.pt')


## plot losses
plotter = Plotter()
plotter.plot_loss(trainer,metrics=['dum_loss','mean_loss','data_loss'],logy=True)

plt.show()