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

import matplotlib.pyplot as plt
from copy import deepcopy

# ----------------------------------------------- 
# ------------ PREPROCESSING DATA ---------------
# ----------------------------------------------- 
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

u_cpy = deepcopy(u)

# sparsifying the input field by keeping every 5 values of u (keeping the first and last as well) and filling the rest with x
u[:,1:5] = L[1:5]
u[:,6:10] = L[6:10]
u[:,11:15] = L[11:15]
u[:,16:20] = L[16:20]
u[:,21:25] = L[21:25]
u[:,26:29] = L[26:29]


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

# ----------------------------------------------- 
# ----------------------------------------------- 


# define the dataset
class TimeSeriesDataset(Dataset):
    def __init__(self,dataTensor) -> None:
        super().__init__()

        self.data = dataTensor

    def __len__(self):
        return len(self.data)-1
    
    def __getitem__(self, index):
        return torch.stack((self.data[index],self.data[index+1]))
        #return self.data[index]

trainData = TimeSeriesDataset(u_train[:,None,:])

trainDataTensor = torch.zeros((len(trainData),2,1,30))
for i in range(len(trainData)):
    trainDataTensor[i,...] = trainData[i]
trainDataTensor = LabelTensor(trainDataTensor,labels)

u0 = torch.tensor(u_cpy[0,:],dtype=torch.float).reshape((1,1,nx))
u0 = LabelTensor(u0,labels)

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
A = A.to(device)
B = B.to(device)

# define the problem
class MOR(AbstractProblem):
    input_variables = labels
    output_variables = labels

    # presumably good enough, still not super accurate for some reason
    # also likely very slow with the for loop. might be able to speed it up 
    # with some pytorch shenanigans
    def physLoss(_input,_output):
        _x = _output.tensor[0,...].squeeze()
        _y = _output.tensor[1,...].squeeze()
        res = torch.zeros((len(_x),nx))
        for i,(x,y) in enumerate(zip(_x,_y)):
            Bloc = x[:,None]*B
            M = A-Bloc
            M[0,0]= 1; M[0,1]=0
            M[-1,-1]= 1; M[-1,-2]=0
            res[i] = (y-x)/dt - torch.matmul(M.detach(),x)
        
        return res
    
    def top_boundary(_input,_output):
        val = _output.tensor[...,-1]
        return val
    
    def bot_boundary(_input,_output):
        val = _output.tensor[...,0]
        return val

    conditions = {'phys':Condition(input_points=trainDataTensor,equation=Equation(physLoss)),
                    't0': Condition(input_points=trainDataTensor[:1,0,:,:],output_points=u0),
                  'x0': Condition(input_points=trainDataTensor,equation=Equation(top_boundary)),
                  'x1': Condition(input_points=trainDataTensor,equation=Equation(bot_boundary))}
    # define the burgers error



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

solver = PINN(problem,
                cae,
                optimizer=torch.optim.Adam,
                optimizer_kwargs={'lr':0.001},
                scheduler=torch.optim.lr_scheduler.StepLR,
                scheduler_kwargs={'gamma':0.5,'step_size':500})

loader = iter(DataLoader(trainData,batch_size=20))
trainer = Trainer(solver,
                batch_size=20,
                max_epochs=5000,
                callbacks = [MetricTracker()],accelerator="auto")
#print(type(next(loader)))
# print(next(loader).size())
#print(cae(next(loader)))

# train
try:
    trainer.train()
except KeyboardInterrupt:
    torch.save(cae,"model_new_sparse.pt")
#torch.save(cae,"model_new.pt")
#torch.save(cae,"model_new_sparse.pt")

pred = cae(torch.tensor(u.reshape((nt,1,nx)),dtype=torch.float)).detach().numpy().reshape((nt,nx))
print(u.shape)
print(pred.shape)

fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(121)
real_plt = ax1.imshow(u_cpy.T,extent=[t[0],t[-1],L[0],L[-1]])
ax1.set_aspect(0.25)
ax1.set_title('Real')
ax1.set_xlabel('t')
fig.colorbar(real_plt)

ax2 = fig.add_subplot(122)
rec_plt = ax2.imshow(pred.T,extent=[t[0],t[-1],L[0],L[-1]])
ax2.set_aspect(0.25)
ax2.set_title('Autoencoder')
ax2.set_xlabel('t')
fig.colorbar(rec_plt)

plt.show()
