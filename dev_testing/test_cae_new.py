"""
Testing the version that uses a custom dataset to train the model.
Possibly also sparsify the input.
"""

# import all the stuff
from pina.problem import AbstractProblem
from pina import LabelTensor
from pina.condition import Condition
from pina.equation import Equation
from pina.trainer import Trainer
from pina.solvers import PINN
from pina.model import CAE
from pina.callbacks import MetricTracker
from torch.utils.data import Dataset,DataLoader

import torch

from scipy.io import loadmat


data = loadmat("Burgers_FOM.mat")
u = data['u_Mat'].T # (nt,nx)
nt = u.shape[0] 
nx = u.shape[1]
L = data['L'] # (nx,1)
L = L.reshape(nx)
t = data['time'] # (nt,1)
t = t.reshape(nt)
dx = L[1]-L[0]
dt = t[1]-t[0]

## getting domain boundaries to define the domain in the PINA problem
x_dom = [L[0], L[-1]]
t_dom = [t[0], t[-1]]

## making L and t LabelTensors as required by PINA
L_tens = LabelTensor(L,'x')
t_tens = LabelTensor(t,'t')

labels = [f'u{i}' for i in range(u.shape[-1])]

## train test split
# train size is 80% of data
train_size = int(0.8*nt)
test_size = nt-train_size


u_train = torch.Tensor(u[:train_size,:])
u_test = torch.Tensor(u[train_size:,:])

u_train = LabelTensor(u_train,labels)
u_test = LabelTensor(u_test,labels)

# define the dataset
class TimeSeriesDataset(Dataset):
    def __init__(self,dataTensor) -> None:
        super().__init__()

        self.data = dataTensor

    def __len__(self):
        return len(self.data)-1
    
    def __getitem__(self, index):
        return torch.cat((self.data[index],self.data[index+1]),0)

trainData = TimeSeriesDataset(u_train[:,None,:])


# matrices for linear and nonlinear part
mu = 0.1
Adiags = [[1 for _ in range(nx-1)], [-2 for _ in range(nx)], [1 for _ in range(nx-1)]]
A = torch.diag(torch.tensor(Adiags[0]),-1)
A += torch.diag(torch.tensor(Adiags[1]),0)
A += torch.diag(torch.tensor(Adiags[2]),1)
A = A*mu/dx**2

Bdiags = [[-1 for _ in range(nx)],[1 for _ in range(nx-1)]]
B = torch.diag(torch.tensor(Bdiags[0]),0)
B += torch.diag(torch.tensor(Bdiags[1]),1)
B = B/dx

# define the problem
class MOR(AbstractProblem):
    input_variables = labels
    output_variables = labels

    # presumably good enough, still not super accurate for some reason
    # also likely very slow with the for loop. might be able to speed it up 
    # with some pytorch shenanigans
    @staticmethod
    def physLoss(_x,_y):
        res = torch.zeros((len(_x),nx))
        for i,(x,y) in enumerate(zip(_x,_y)):
            Bloc = x[:,None]*B
            M = A-Bloc
            M[0,0]= 1; M[0,1]=0
            M[-1,-1]= 1; M[-1,-2]=0
            res[i] = (y-x)/dt - torch.matmul(M.detach(),x)
        
        return res
    
    conditions = {'phys':Condition(input_points=u_train,equation=Equation(physLoss))}
    # define the burgers error


# loader = iter(DataLoader(trainData,batch_size=2))

problem = MOR()

# define the model
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

solver = PINN(problem,cae)

trainer = Trainer(solver,batch_size=20,max_epochs=1000,callbacks = [MetricTracker()],accelerator="auto")
print(trainData[0].size())
# print(cae(trainData[0]))

# train
# trainer.train()
# torch.save(cae,"model_new.pt")

