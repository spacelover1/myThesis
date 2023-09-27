# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 02:15:03 2020

@author: AMIN
"""
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, MaxPooling1D, Dropout
from keras.optimizers import Adam
from keras import callbacks
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
import os

def prepare(data):
    all_data = []
    labels = []
    for i in range(np.shape(data)[0]):
        for j in range(np.shape(data[i])[0]):
            all_data.append(list(data[i][j]))
            labels.append(i)
            
    return np.array(all_data), np.reshape(np.array(labels), (-1, 1))

def load_dataset(address, step, split):
    train = []
    test = []
    for file in os.listdir(address):
        arr = np.array(pd.read_csv(address+file))
        s = int(np.shape(arr)[0]/step) * step
        arr = np.reshape(arr[0:s, 0], (-1, step, 1)) 
        n_samples = np.shape(arr)[0]
        n_train = int(n_samples*split)
        train_idx = np.random.choice(range(0, n_samples), size=n_train, replace=False)
        test_idx = list(set(range(0,n_samples))-set(train_idx))
        train.append(arr[train_idx])
        test.append(arr[test_idx])
    
    trainX, trainy = prepare(train)
    testX, testy = prepare(test)
       	
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    
    return trainX, trainy, testX, testy

def evaluate_model(trainX, trainy, testX, testy):
    
    verbose, epochs, batch_size = 1, 20, 10000
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    
    model = Sequential()

    model.add(LSTM(units = 50, return_sequences = True, input_shape = (n_timesteps, n_features)))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(units = n_outputs, activation='softmax'))
   
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    
    model.summary()
    
    #X1, X2, Y1, Y2 = train_test_split(trainX, trainy, test_size=0.25, random_state=42) 
    
    #reducelronplateau = callbacks.ReduceLROnPlateau(monitor='val_loss',
    #                                                factor=0.7,
    #                                                patience=2)
    
    #callbacks_list = [reducelronplateau]
    
    # fit network
    #model.fit(X1, Y1, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks = callbacks_list, validation_data=(X2, Y2))
	
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=verbose)
    
    return accuracy



def summarize_results(scores):
	print(scores)
	m, s = np.mean(scores), np.std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
 
# run an experiment
def run_experiment(repeats=1):
	# load data
    trainX, trainy, testX, testy = load_dataset(address='./csv files/', step=100, split=0.8)
	# repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(trainX, trainy, testX, testy)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)
 
# run the experiment
run_experiment()

