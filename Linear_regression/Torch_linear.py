# %%
import numpy as np 
import torch 
import matplotlib.pyplot as plt 

import torch.nn.functional as F
from torch import nn

# %%
n = 100

d = 2
x = np.random.uniform(-1, 1, (n, d))

# y = 5x + 10
weights_true = np.array([[5],[5]])
bias_true = np.array([10])

y_true = x @ weights_true + bias_true
plt.scatter(x[:,0],y_true)


# %%
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x[:,0],x[:,1],y_true)



# %%
x_torch=torch.tensor(x,dtype=torch.float32)
y_torch=torch.tensor(y_true,dtype=torch.float32)

# %%
class Model(nn.Module):
    
    def __init__(self, in_size,units=1):
        super(Model,self).__init__()
        # hidden layer
        self.linear1 = nn.Linear(in_size, units)
        
   
    def training_step(self,x_train,y_true):
        y_pred=self.linear1(x_train)
        loss=F.mse_loss(y_pred,y_true)
        
        return loss
    def epoch_end(self, epoch, train_loss):
        print("Epoch [{}], _loss: {:.4f}, ".format(epoch, train_loss))
    
       
      

# %%
model=Model(2)
model.training_step(x_torch,y_torch)

# %%
def fit(x,y,epochs,lr, model,opt_func=torch.optim.SGD):

    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        
        loss = model.training_step(x,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Validation phase
        if epoch%10==0:
        
            model.epoch_end(epoch, loss.detach().numpy())
        history.append(loss.detach().numpy())
    return history

# %%
history=fit(x_torch,y_torch,200,0.01,model)

# %%
plt.plot(history)

# %%



