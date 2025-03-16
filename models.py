# MORE ABOUT THIS FILE:

# All the models used for the multi-task approach


import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout, Activation, Input, Concatenate, Lambda
from keras import backend as K
import numpy as np
import pandas as pd
from tensorflow.keras import layers


def nn_binary_clasification(dim, lay, activation):
    model = keras.Sequential()
    model.add(Input(shape=(dim,)))
    if lay:        
        for units in lay:
            model.add(Dense(units, activation=activation))
 
    model.add(Dense(1, activation='sigmoid'))
    return model


def nn_reg1(dim, lay, activation):
    model = keras.Sequential()
    model.add(Input(shape=(dim,)))
    if lay:        
        for units in lay:            
            model.add(Dense(units, activation=activation))  

    model.add(Dense(1, activation='linear'))
    return model


def nn_reg_n(dim, lay, activation, n):
    model = keras.Sequential()
    model.add(Input(shape=(dim,)))
    if lay:
        for units in lay:      
            model.add(Dense(units, activation=activation))  
    model.add(Dense(n, activation='linear'))
    return model


#External layer
class ExtLayer(layers.Layer):
    def __init__(self, initial_sigma=0.1):
        super(ExtLayer, self).__init__()
        # Initialize the trainable variable
        self.var = tf.Variable(initial_value=[[initial_sigma]], dtype=tf.float32, trainable=True)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]  # Get the current batch size
        # Tile sigma to match the batch size
        tensor = tf.tile(self.var, [batch_size, 1])
        return tensor



def mt2(dim, lay_clas, lay_reg, acti):
    #Regresion model
    input_layer = keras.Input(shape=(dim,), name = 'reg')  
    # Define shared hidden layers
    if lay_reg: 
        hidden = layers.Dense(lay_reg[0], activation= acti)(input_layer)
        for units in lay_reg[1:]:
            hidden = layers.Dense(units, activation=acti)(hidden)
        shared_output = hidden
    else:
        shared_output = input_layer
        
    reg = layers.Dense(1, activation='linear', name='regression_output')(shared_output)

    sigma_output = ExtLayer()(input_layer)

    concatenate = tf.stack([reg, sigma_output], axis = 1)
    mod_reg = keras.Model(inputs=[input_layer], outputs=[concatenate])
    pre_pi = mod_reg(input_layer)
    pre_reg, pre_sigma = tf.unstack(pre_pi, num=2, axis=1)

    #Clasification model

    input_clas = keras.Input(shape=(dim +1 ,), name = 'clas')  
    # Define shared hidden layers
    if lay_clas: 
        hidden = layers.Dense(lay_clas[0], activation= acti)(input_clas)
        for units in lay_clas[1:]:
            hidden = layers.Dense(units, activation=acti)(hidden)
        shared_output_c = hidden
    else:
        shared_output_c = input_clas

    clas = layers.Dense(1, activation='sigmoid', name='classification_output')(shared_output_c)
    
    temp_output = ExtLayer()(input_clas)

    concatenate_clas = tf.stack([clas, temp_output], axis = 1)
    mod_clas = keras.Model(inputs=[input_clas], outputs=[concatenate_clas])
    combined_input = layers.Concatenate()([input_layer, pre_reg])
    pre_prob = mod_clas(combined_input)
    pre_clas, pre_temp = tf.unstack(pre_prob, num=2, axis=1)


    concatenate_f = tf.stack([pre_clas, pre_reg, pre_sigma, pre_temp], axis = 1)
    model = keras.Model(inputs=[input_layer], outputs=[concatenate_f])
    return model





def mt2_n(dim, lay_clas, lay_reg, acti, n):
    #Regresion model
    input_layer = keras.Input(shape=(dim,), name = 'reg')  
    # Define shared hidden layers
    if lay_reg: 
        hidden = layers.Dense(lay_reg[0], activation= acti)(input_layer)
        for units in lay_reg[1:]:
            hidden = layers.Dense(units, activation=acti)(hidden)
        shared_output = hidden
    else:
        shared_output = input_layer

    out_reg = []
    for i in range(n):
        reg = layers.Dense(1, activation='linear', name=f'regression_output_{i}')(shared_output)
        out_reg.append(reg)
    for i in range(n):
        sigma = ExtLayer()(input_layer)
        out_reg.append(sigma)

    concatenate = tf.stack(out_reg, axis = 1)
    mod_reg = keras.Model(inputs=[input_layer], outputs=[concatenate])
    pre_pi = mod_reg(input_layer)

    SS = tf.unstack(pre_pi, num=2*n, axis=1)
    pre_reg = [tf.reshape(SS[i], [-1,1]) for i in range(n)]
    pre_sigma = [tf.reshape(SS[i + n], [-1,1]) for i in range(n)]

    #Clasification model

    input_clas = keras.Input(shape=(dim + n,), name = 'clas')  
    # Define shared hidden layers
    if lay_clas: 
        hidden = layers.Dense(lay_clas[0], activation= acti)(input_clas)
        for units in lay_clas[1:]:
            hidden = layers.Dense(units, activation=acti)(hidden)
        shared_output = hidden
    else:
        shared_output = input_clas

    clas = layers.Dense(1, activation='sigmoid', name='classification_output')(shared_output)
    temp_output = ExtLayer()(input_clas)

    concatenate_clas = tf.stack([clas, temp_output], axis = 1)
    mod_clas = keras.Model(inputs=[input_clas], outputs=[concatenate_clas])


    combined_input = layers.Concatenate()([input_layer, pre_reg[0]])
    for i in range(1, n):
        combined_input = layers.Concatenate()([combined_input, pre_reg[i]])
    pre_prob = mod_clas(combined_input)
    pre_clas, pre_temp = tf.unstack(pre_prob, num=2, axis=1)

    out = [pre_clas]
    for i in range(n):
        out.append(pre_reg[i])
    for i in range(n):
        out.append(pre_sigma[i])
    out.append(pre_temp)

    concatenate_f = tf.stack(out, axis = 1)
    model = keras.Model(inputs=[input_layer], outputs=[concatenate_f])
    return model


