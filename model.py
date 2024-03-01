import os
import datetime
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import tensorflow as tf
import wandb 
import cp_detection_timeseries as cpdt
import random

mpl.rcParams["figure.figsize"] = (6, 3)
mpl.rcParams["axes.grid"] = False


import Data_Loader




def get_model(model_name):
    model = MODELS.get(model_name)
    if model is None:
        raise NotImplementedError("This model name is not included yet. Please choose from 'last_baseline', 'linear', 'dense', 'conv', 'lstm', or 'AR lstm'.")
    return model

       
# train.py
#from models import get_model
#def main():
#    import argparser






def compile_and_fit(model, window, patience=5, MAX_EPOCHS=40): #lstm_units, out_steps, num_features,
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, mode="min"
    )

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )

    history = model.fit(
        window.train,
        epochs=MAX_EPOCHS,
        validation_data=window.val,
        callbacks=[early_stopping],
        verbose=0
    )
    return history

#def evaluate_model(model, window):
#    val_performance[model.name] = model.evaluate(window.val)
#    performance[model.name] = model.evaluate(window.test, verbose=0)

def plot_model_performance(model, window):
    window.plot(model)
    


class MultiStepLastBaseline(tf.keras.Model):
    def call(self, inputs):
        return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])

class FeedBack(tf.keras.Model):
    def __init__(self, units, num_features, out_steps):
            super().__init__()
            self.out_steps = out_steps
            self.units = units
            self.lstm_cell = tf.keras.layers.LSTMCell(units)
            self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
            self.dense = tf.keras.layers.Dense(num_features)

    def warmup(self, inputs):
        x, *state = self.lstm_rnn(inputs)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
        predictions = []
        prediction, state = self.warmup(inputs)
        predictions.append(prediction)
        for n in range(1, self.out_steps):
            x = prediction
            x, state = self.lstm_cell(x, states=state, training=training)
            prediction = self.dense(x)
            predictions.append(prediction)
        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions

# models.py    
    
MODELS = {}   

# Define a placeholder class for models
class ModelPlaceholder:
    pass

# Initialize the dictionary with placeholders
MODELS = {
    "last_baseline": ModelPlaceholder(),
    "linear": ModelPlaceholder(),
    "dense": ModelPlaceholder(),
    "conv": ModelPlaceholder(),
    "lstm": ModelPlaceholder(),
    "AR lstm": ModelPlaceholder()
}


"""
def get_model_new(model_kind, out_steps, num_features, **kwargs):
    if model_kind == "asd":
        model = _build_smothing_function(model_kind, out_steps)
    elif model_kind == "super":
        extra_special = kwargs.pop("extra")
        if extra_special is None:
            raise Exception("need to provide extr a
                            
                
    
    print(kwargs)
    
get_model_new(12, 12, 14, fish=42)
"""

# Define specific models and assign them to the dictionary keys
def define_models(OUT_STEPS, num_features, CONV_WIDTH=3, lstm_units=100):
    MODELS["last_baseline"] = MultiStepLastBaseline()

    linear_model = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        tf.keras.layers.Dense(OUT_STEPS*num_features, kernel_initializer=tf.initializers.zeros()),
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])
    
    MODELS["linear"] = linear_model

    dense_model = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(OUT_STEPS*num_features, kernel_initializer=tf.initializers.zeros()),
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])
    
    MODELS["dense"] = dense_model

    conv_model = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
        tf.keras.layers.Conv1D(256, activation='relu', kernel_size=CONV_WIDTH),
        tf.keras.layers.Dense(OUT_STEPS*num_features, kernel_initializer=tf.initializers.zeros()),
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])
    
    MODELS["conv"] = conv_model
    
    lstm_model = tf.keras.Sequential([
        tf.keras.layers.LSTM(lstm_units, return_sequences=False),
        tf.keras.layers.Dense(OUT_STEPS*num_features, kernel_initializer=tf.initializers.zeros()),
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])
    
    MODELS["lstm"] = lstm_model
    
    AR_lstm_model = FeedBack(units=lstm_units, num_features=num_features, out_steps=OUT_STEPS)
    
    MODELS["AR lstm"] = AR_lstm_model
    return
   


def plot_variable_importance(lstm_model, window):
    lstm_weights = lstm_model.layers[0].get_weights()[0]
    normalized_weights = lstm_weights / np.sum(np.abs(lstm_weights), axis=0)
    variable_importance = np.sum(normalized_weights, axis=1)

    variable_importance_and_names_sorted = sorted(zip(variable_importance, window.input_columns))
    variable_importance_sorted, names_sorted = zip(*variable_importance_and_names_sorted)
    
    plt.figure(figsize=(4,4), dpi=100)
    plt.barh(np.arange(len(variable_importance)), variable_importance_sorted, color='teal')
    plt.yticks(np.arange(len(variable_importance)), names_sorted)
    plt.xlabel('Relative Importance Score')
    plt.ylabel('Input Variables')
    plt.show()



def plot_loss(history):
    fig=plt.figure(figsize=(10,4), dpi=100)
    #plt.title(f'{model}: input {IN_WIDTH} time steps, output {OUT_STEPS} timesteps')
    color = iter(cm.tab20c(np.linspace(0, 1, 5)))
    c =next(color)
    plt.plot(history.history['loss'], marker=".", label='train', color=c)
    plt.plot(history.history['val_loss'], label='val', color=c, ls='--')

    plt.yscale('log')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend()
    #legend = plt.legend(title="stations used to predict T2 in dakar",
                        #loc=1, fontsize='small', fancybox=True)

        
        
        
