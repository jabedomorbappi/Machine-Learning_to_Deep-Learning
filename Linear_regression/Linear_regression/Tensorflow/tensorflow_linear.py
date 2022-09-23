# %%
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 


# %%
# number of training samples to create
n_samples = 10000
# `m` and `c` are coefficient and bias to get the initial `y`
m = 9
c = -2
mean = 0.0 # mean of the training data distribution to create
std = 1.0 # standard deviation of the of the training data distribution to create
# number of training epochs 
num_epochs = 3000
# learning rate
learning_rate = 0.001

# %%
def create_dataset(n_samples, m, c):
    # create the sample dataset
    x = np.random.normal(mean, std, n_samples)
    random_noise = np.random.normal(mean, std, n_samples)
    y = m*x + c + random_noise
    x_train, y_train = x[:8000], y[:8000]
    
    
    x_test, y_test = x[8000:], y[8000:]
   
    
    return x_train, y_train, x_test, y_test

# %%
x_train, y_train, x_test, y_test=create_dataset(n_samples,m,c)

print(f"Training samples: {len(x_train)}")
print(f"Test samples: {len(x_test)}")
plt.figure(figsize=(12, 9))
plt.scatter(x_train, y_train)
plt.xlabel('x_train')
plt.ylabel('y_train')
plt.title('Training data distribution')
plt.savefig('training_data.jpg')

# %%
x_train=np.array(x_train).reshape((x_train.shape[0],1))
x_test=np.array(x_test).reshape((x_test.shape[0],1))
y_train=np.array(y_train).reshape((y_train.shape[0],1))
y_test=np.array(y_test).reshape((y_test.shape[0],1))

# %%
#W = tf.Variable(np.random.randn())
#B = tf.Variable(np.random.randn())
##print(W)
#print(B)

# %%
class LinearModel(tf.keras.Model):
    
    def __init__(self,input_dim,units=1):
        super(LinearModel,self).__init__()
        self.weight=tf.Variable(np.random.randn(input_dim,units),trainable=True)
        self.bias=tf.Variable(np.zeros(units),trainable=True)
        
        #self.weight = self.add_weight(shape=(input_dim, units),initializer="random_normal",trainable=True)
        #self.bias = self.add_weight(shape=(units,), initializer="random_normal", trainable=True)

    def forward(self,xb):
        self.y_pred=tf.matmul(xb, self.weight) + self.bias 
        
        return  self.y_pred  
    def mse(self,y_true):
        self.loss=tf.reduce_mean(tf.square(tf.subtract(y_true,self.y_pred)))
        
        return self.loss 
    
    
    


# %%
model=LinearModel(x_train.shape[1])
y_pred=model.forward(x_train)
loss=model.mse(y_train)




# %%
def fit():
    for epoch in range(100):
        with tf.GradientTape() as tape:
            y_pred=model.forward(x_train)
        
            loss=model.mse(y_train)
        grads=tape.gradient(loss,[model.weight,model.bias])
        model.weight.assign_sub(grads[0]*0.01)
        model.bias.assign_sub(grads[1]*0.01)
        if epoch%10==0:
            print('epoch {} loss {}'.format(epoch,loss.numpy()))

    
  


# %%
fit()

# %%
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers


import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

# %%
inputs = keras.Input(shape=(x_train.shape))
#l1 = layers.Dense(10, activation='relu', name='dense_1')(inputs)
outputs = layers.Dense(1, name='regression')(inputs)

model = keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())
model.compile(loss='mse', optimizer=optimizers.SGD(0.1))

model.fit(x_train,y_train, epochs=10,batch_size=32)






# %%
x_train.shape
x.shape

# %%
n = 100

d = 2
x = np.random.uniform(-1, 1, (n, d))

# y = 5x + 10
weights_true = np.array([[5],[5]])
bias_true = np.array([10])

y_true = x @ weights_true + bias_true
#plt.scatter(x[:,0],y_true)


inputs = keras.Input(shape=(x.shape[1],))
#l1 = layers.Dense(10, activation='relu', name='dense_1')(inputs)
outputs = layers.Dense(1, name='regression')(inputs)

model = keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())
model.compile(loss='mse', optimizer=optimizers.SGD(0.1))




# %%
model.fit(x,y_true, epochs=10)

# %%



