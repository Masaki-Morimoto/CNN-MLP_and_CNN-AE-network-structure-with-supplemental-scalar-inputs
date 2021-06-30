'''
CNN-MLP model with scalar input

This file contains a code for forming a network structure of the CNN-MLP model used for force coefficient estimation of flow over a flat plate (illustration: fig. 3, description: table 1)

Especially, an exmaple for 'Case 1' is shown

It can be easily modified for the other cases by taking out upsampling layers from the sub-network and replacing the concatenating layer in the mainstream network

Configurations of the input & output varaiables are as follows:
input_img - vorticity field of flow around an inclined airfoil: nx = 147, ny = 131
input_sc - scalar values: Re number and AoA
output - drag and lift coefficients
'''

# To run this code, a listed modules are required
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.backend import tf as ktf
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Concatenate, Lambda

# input variables
input_img = Input(shape=(nx,ny,1)) # vorticitiy field with its size of (number of snapshots, nx, ny, 1)
input_sc = Input(shape=(2,))       # scalar values (Re number and AoA) with its size of (number of snapshots, 2)

# network parameters
fs = 7       # filter size for the convolutional layers
act = 'relu' # activation function

#-- [sub-network] MLP-CNN part for extending scalar values to two-dimensional data --#
xsc = Dense(16, activation=act)(input_sc)
xsc = Dense(32, activation=act)(xsc)
xsc = Dense(64, activation=act)(xsc)
xsc = Dense(256, activation=act)(xsc)
xsc = Dense(512, activation=act)(xsc)
xsc = Dense(1024, activation=act)(xsc)
xsc = Dense(1221, activation=act)(xsc)
xsc = Reshape([37,33,1])(xsc)
xsc = Conv2D(2, (fs,fs), activation=act, padding='same')(xsc)
xsc = UpSampling2D((2,2))(xsc) # (74,66)
xsc = Conv2D(2, (fs,fs), activation=act, padding='same')(xsc)
xsc = UpSampling2D((2,2))(xsc) # (148,132)
xsc = Conv2D(2, (fs,fs), activation=act, padding='same')(xsc)

#-- [mainstraem network] CNN-MLP part --#
x = Lambda(lambda image: K.tf.image.resize_images(image,(148,132)))(input_img) # input vorticity field
x = Conv2D(16, (fs,fs), activation=act, padding='same')(x) # (148,132) 
x = Concatenate()([x,xsc]) # concatenate 'xsc' (two-dimensional data from sub-network) and 'x' (vorticity field)
x = MaxPooling2D((2,2))(x) # (74,66)
x = Conv2D(16, (fs,fs) ,activation=act, padding='same')(x)
x = MaxPooling2D((2,2))(x) # (37,33)
x = Conv2D(16, (fs,fs), activation=act, padding='same')(x)
x = Conv2D(8, (fs,fs), activation=act, padding='same')(x)
x = Conv2D(4, (fs,fs), activation=act, padding='same')(x)
x = Conv2D(1, (fs,fs), activation=act, padding='same')(x)
x1 = Reshape([37*33*1])(x)
x1 = Dense(1024, activation=act)(x1)
x1 = Dense(256, activation=act)(x1)
x1 = Dense(64, activation=act)(x1)
x1 = Dense(32, activation=act)(x1)
x1 = Dense(16, activation=act)(x1)
x_final = Dense(2, activation='linear')(x1)


model = Model([input_img,input_sc], x_final)
model.compile(optimizer='adam', loss='mse')