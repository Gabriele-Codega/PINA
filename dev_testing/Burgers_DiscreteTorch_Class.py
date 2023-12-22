"""
Author: Rahul Halder
"""

import numpy as np
import torch

## IMPORTANT: check correctness of the results.
# should be correct, checked by evaluating the residual on the training set
# with no pass to the autoencoder.
class Burgers_Discrete:
     
     def __init__(self,L,T,mu,um1,up1):
     
         self.nx = len(L)
         self.nt = len(T)
         self.mu = mu
         self.L = L
         self.T = T
         # modify to accept batched input
         self.batch_size = um1.size(dim=0)
         self.um1 = um1.reshape((self.batch_size,self.nx))
         self.up1 = up1.reshape((self.batch_size,self.nx))
         self.dx = self.L[1]-self.L[0]
         self.dt = self.T[1]-self.T[0] 
         self.const = (self.mu/(self.dx*self.dx))
         
     def Compute_LinearMat(self):
          A =  np.zeros((self.nx, self.nx)) 
          for i in range(1,self.nx-1):
              if (i == 1) :
                  A[i,i+1] = 1
              elif (i == self.nx-2) :
                  A[i,i-1] = 1
              else:
                  A[i,i+1]=1
                  A[i,i-1]=1
              A[i,i] = - 2
          A_tensor = torch.from_numpy(A).float()
          return A_tensor
      
     def Compute_NonlinearMat(self):
         # one matrix F for each vector in batch
         F = torch.zeros((self.batch_size,self.nx, self.nx))
         for i in range(1,self.nx-1):
             
             if (i == self.nx-2) :
                F[:,i,i] = -self.um1[:,i]
             else:
                F[:,i,i+1]= self.um1[:,i]
                F[:,i,i]=  -self.um1[:,i]
         
         return F

     def Burgers_Residual_Temp(self):
         return (1/self.dt)*(self.up1-self.um1)
     
     def Burgers_Residual_Spatial(self):
         Alin =  self.Compute_LinearMat()
         Fnonlin = self.Compute_NonlinearMat()
         # using einsum to perform matrix products in batches
         return - self.const*torch.einsum('ij,bj->bi',Alin,self.um1)+(1.0/self.dx)*torch.einsum('bij,bj->bi',Fnonlin,self.um1)
         
      
     def Burgers_Residual(self):
         return self.dx*(self.Burgers_Residual_Temp() + self.Burgers_Residual_Spatial())