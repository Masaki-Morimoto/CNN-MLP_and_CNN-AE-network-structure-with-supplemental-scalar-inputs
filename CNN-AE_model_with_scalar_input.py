'''
CNN-AE model with scalar input

This file contains a code for forming a network structure of the CNN-AE model used for the low-dimensionalization of two-dimensional isotopic homogeneous turnulence (illustration: fig. 4, description: table 3)

Especially, an exmaple for 'Case 1' is shown

It can be easily modified for the other cases by taking out upsampling layers from the sub-network and replacing the concatenating layer in the mainstream network

Configurations of the input & output varaiables are as follows:
input_img - vorticity field of the flow: nx = 256, ny = 256
input_re - scalar values: initla Re number
output - input vorticity field
'''

# To run this code, a listed modules are required
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.backend import tf as ktf
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Concatenate, Lambda


# input variables
nput_img = Input(shape=(256,256,1)) # vorticitiy field with its size of (number of snapshots, nx, ny, 1)
input_re = Input(shape=(2,))        # scalar values (Re number) with its size of (number of snapshots, 2)

# network parameters
num_channel=32 # number of the filters for the covolutional layers
fs = 7         # filter size for the convolutional layers
act = 'relu'   # activation function
#-------------------#

#-- [sub-network] MLP-CNN part for extending scalar values to two-dimensional data --#
xsc = Dense(16, activation=act)(input_re)
xsc = Dense(32, activation=act)(xsc)
xsc = Dense(64, activation=act)(xsc)
xsc = Dense(256, activation=act)(xsc)
xsc = Reshape([16,16,1])(xsc)
xsc = Conv2D(2, (fs,fs), activation=act, padding='same')(xsc)
xsc = UpSampling2D((2,2))(xsc) #(32,32)
xsc = Conv2D(2, (fs,fs), activation=act, padding='same')(xsc)
xsc = UpSampling2D((2,2))(xsc) #(64,64)
xsc = Conv2D(2, (fs,fs), activation=act, padding='same')(xsc)
xsc = UpSampling2D((2,2))(xsc) #(128,128)
xsc = Conv2D(2, (fs,fs), activation=act, padding='same')(xsc)
xsc = UpSampling2D((2,2))(xsc) #(256,256)
xsc = Conv2D(2, (fs,fs), activation=act, padding='same')(xsc)
xsc = Conv2D(2, (fs,fs), activation=act, padding='same')(xsc)


#-- [mainstraem network] CNN-AE part --#
x = Conv2D(num_channel,(fs,fs),activation=act, padding='same')(input_img) #256,256
x = Concatenate()([x,xsc]) # concatenate 'xsc' (two-dimensional data from sub-network) and 'x' (vorticity field)
x = MaxPooling2D((2,2))(x) #128,128
x = Conv2D(num_channel, (fs,fs), activation=act, padding='same')(x) #128,128
x = MaxPooling2D((2,2))(x) #64,64
x = Conv2D(num_channel, (fs,fs), activation=act, padding='same')(x) #64,64
x = MaxPooling2D((2,2))(x) #32,32
x = Conv2D(num_channel, (fs,fs), activation=act, padding='same')(x) #32,32
x = MaxPooling2D((2,2))(x) #16,16
x = Conv2D(num_channel, (fs,fs), activation=act, padding='same')(x) #16,16
x = MaxPooling2D((2,2))(x) #8,8
x = Conv2D(num_channel, (fs,fs), activation=act, padding='same')(x) #8,8
x = Conv2D(8,(fs,fs), activation=act, padding='same')(x)
x = Conv2D(4,(fs,fs), activation=act, padding='same')(x)
x = Conv2D(2,(fs,fs), activation=act, padding='same')(x)
lat_space = Conv2D(2,(fs,fs),activation=act, padding='same')(x)
x = Conv2D(2,(fs,fs), activation=act, padding='same')(lat_space)
x = Conv2D(4,(fs,fs), activation=act, padding='same')(x)
x = Conv2D(8,(fs,fs), activation=act, padding='same')(x)
x = Conv2D(num_channel, (fs,fs), activation=act, padding='same')(x) #8,8
x = UpSampling2D((2,2))(x) #16,16
x = Conv2D(num_channel, (fs,fs), activation=act, padding='same')(x) #16,16
x = UpSampling2D((2,2))(x) #32,32
x = Conv2D(num_channel, (fs,fs), activation=act, padding='same')(x) #32,32
x = UpSampling2D((2,2))(x) #64,64
x = Conv2D(num_channel, (fs,fs), activation=act, padding='same')(x) #64,64
x = UpSampling2D((2,2))(x) #128,128
x = Conv2D(num_channel, (fs,fs), activation=act, padding='same')(x) #128,128
x = UpSampling2D((2,2))(x) #256,256
x = Conv2D(num_channel, (fs,fs), activation=act, padding='same')(x) #256,256
x = Conv2D(1,(fs,fs), activation='linear', padding='same')(x) #256,256
#-------------------#
model = Model([input_img,input_re], x)
model.compile(optimizer='adam', loss='mse')