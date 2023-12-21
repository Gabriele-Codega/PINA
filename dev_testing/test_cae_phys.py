"""
   Testing the convolutional autoencoder structure 
   in the PINA framework (v. 0.1).
   The model is a Burgers equation in 1D.

"""

import torch
from Burgers_DiscreteTorch_Class import Burgers_Discrete
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
L_tens = LabelTensor(L,'x')
t_tens = LabelTensor(t,'t')

## train test split
# train size is 80% of data
train_size = int(0.8*nt)
test_size = nt-train_size
gen = torch.Generator()
# setting a seed for reproducibility
gen.manual_seed(1)
train,test = torch.utils.data.random_split(range(nt),[train_size,test_size],generator=gen)
u_train = torch.Tensor(u[train.indices,:])
u_test = torch.Tensor(u[test.indices,:])

## rehaping u to have 1 channel, as required by Conv1D
## also making u a LabelTensor as reuqired by PINA
u = u.reshape((nt,1,nx))
u_train = u_train.reshape((train_size,1,nx))
u_test = u_test.reshape((test_size,1,nx))
u_stack = torch.stack([u_train[:train_size-1,...],u_train[1:,...]],dim=1)
labels = [f'u{i}' for i in range(u.shape[-1])]
u = LabelTensor(u,labels)
u_train = LabelTensor(u_train,labels)
u_test = LabelTensor(u_test,labels)
u_stack = LabelTensor(u_stack,labels)



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


    def Burgers(input_,output_):
        ## BEWARE: crazy shenanigans here!
        # What happens here is extremely inefficient and hence unbearably slow.
        # problem is that we want to work with batched input but the Burgers_Discrete class does not
        # support any kind of batches. The solution is to iterate over the elements of the batch
        # and compute the residual for each of them, after converting them from LabelTensor to Tensor. 
        # The residuals are stored in a list which is then converted to a tensor and returned.
        # The reason is that then PINA uses the return value of the function to 
        # compute the solver's loss, which requires a tensor as input.
        # I am not even 100% sure this is working and moreover it probably won't work for unbatched input
        # (see the CAE class itself for more sketchy code)
        shape = list(output_.size())
        n = shape[0]
        res_ = []
        for i in range(n):
            bd = Burgers_Discrete(L,t,0.1,output_[i,0,...].tensor,output_[i,1,...].tensor)
            res = bd.Burgers_Residual()
            res_.append(res)
        batch_residual = torch.stack(res_,dim=0)
        return batch_residual

    # We can pass points from the spatial/temporal/spatio-temporal domain
    # as LabelTensors with something like `Condition(input_points=pts, equation=Burgers)`.
    # This way we do not actually need to sample the domains and we do not necessarily 
    # have to work with the pairs (x,t) which would come from PINA internals.

    conditions = {'phys': Condition(input_points=u_stack,equation=Equation(Burgers)),
                  't_0': Condition(input_points=u[:1,:,:],output_points=u[:1,:,:])}

rom = PIMOR()


solver = PINN(problem=rom,
              model=cae,
              optimizer=torch.optim.Adam,
              optimizer_kwargs={'lr':0.001},
              scheduler_kwargs={'factor':0.5})
trainer = Trainer(solver,batch_size=20,max_epochs=150,callbacks = [MetricTracker()])


trainer.train()
torch.save(cae, 'pinn_cae_burger_loss.pt')


## plot losses
plotter = Plotter()
plotter.plot_loss(trainer,label='mean_loss',logy=True)

plt.show()