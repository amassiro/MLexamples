import matplotlib.pyplot as plt
import numpy as np


#
# definition of the function (x,y) -> z
#
def theFunction(x,y):
    return np.exp(-0.2*x) * ( 3.5 * np.sin(2*x) + 0.2 * np.sin(20*x) ) + 3*y + np.sin(y)
  

#
# data
#

# x1 and y1 are random values
N=800
x1 = 1.0 * np.random.rand(N)
y1 = 1.0 * np.random.rand(N)

# the z is the function(x1,y1)
z = theFunction(x1,y1) 

# now add all the inputs into one object
all_inputs = np.stack((x1,y1), axis=-1) 


# just plot
plt.figure()
plt.scatter(all_inputs[:,0], all_inputs[:, 1], c=z, cmap=plt.cm.RdBu, edgecolors='k')
        

#
# now keras
# 

from keras.layers import Input, Dense, Dropout, Flatten, LeakyReLU, Add, Concatenate, Dot
from keras.models import Model
from keras.utils  import plot_model
from keras.layers import Layer


#
#                  "2 "because (x,y) are the inputs, thus dimension = 2
inputs = Input(shape=(2,))

# one layer with 500 neurons, and activation function = relu
hidden = Dense(500, activation='relu')(inputs)
# second layer with 100 neurons, and activation function = relu
hidden = Dense(100, activation='relu')(hidden)
outputs = Dense(1)(hidden)
model = Model ( inputs=inputs, outputs=outputs )
model.compile( 
     loss='MSE',
     optimizer='adam'
     )

# wirte the summary of the network
model.summary()

# plot the network
plot_model(
    model,
    to_file="model.png",
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",
)





# This time we split the all_inputs,y upfront
# from sklearn.model_selection import train_test_split
# all_inputs_training, all_inputs_validation, y_training, y_val = train_test_split(all_inputs,y,test_size=0.5) 
nsplit=int(N/2)
all_inputs_training   = all_inputs[:nsplit,:]
all_inputs_validation = all_inputs[nsplit:,:]
z_training   = z[:nsplit]
z_validation = z[nsplit:]

# now actually performing the train 
history = model.fit( all_inputs_training, z_training, validation_data = (all_inputs_validation,z_validation), epochs=150, verbose=0)

# ... and plot the training loss
plt.plot( history.history["val_loss"] )
plt.plot( history.history["loss"] )
plt.show()


# now test the performance of the DNN
z_predicted_validation = model.predict(all_inputs_validation)


plt.plot(all_inputs_validation, z_validation, "b .")
plt.plot(all_inputs_validation, z_predicted_validation, "r +")

plt.show()




