# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:04:26 2020

@author: dilpreet
"""

import os
import pickle
from math import ceil

import numpy as np
import pandas as pd
from kapre.composed import get_melspectrogram_layer
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm


class Network:
    def __init__(self, dataset, class_dist, prob_dist, time=1, n_class=6, rate=16000, mode="cnn1D"):
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),os.path.join("saved_models",mode+".model"))
        self.p_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),os.path.join("pickles",mode+".p"))
        self.history = os.path.join(os.path.dirname(os.path.abspath(__file__)),os.path.join("history",mode+".csv"))
        self.time = time
        self.rate = rate
        self.n_class = n_class
        self.mode = mode

        if self.mode == "cnn1d":
            self.build_rand_feat(dataset, class_dist, prob_dist)
            self.model = self.get_conv1D_model(n_class,rate,time)
        elif self.mode == "cnn2d":
            self.build_rand_feat(dataset, class_dist, prob_dist)
            self.model = self.get_conv2D_model(n_class,rate,time)
        elif self.mode == "rnn":
            self.build_rand_feat(dataset, class_dist, prob_dist)
            self.model = self.get_recurrent_model(n_class,rate,time)
        else:
            return ValueError("invalid mode")
        
        
    def get_conv1D_model(self,N_CLASSES=6, SR=16000, DT=1.0):
        # input shape (n, feat, channel)
        input_shape = (int(SR*DT), 1)
        i = get_melspectrogram_layer(input_shape=input_shape,
                                    n_mels=128,
                                    pad_end=True,
                                    n_fft=512,
                                    win_length=400,
                                    hop_length=160,
                                    sample_rate=SR,
                                    return_decibel=True,
                                    input_data_format='channels_last',
                                    output_data_format='channels_last')
        x = layers.LayerNormalization(axis=2, name='batch_norm')(i.output)
        x = layers.TimeDistributed(layers.Conv1D(8, kernel_size=(4), activation='tanh'), name='td_conv_1d_tanh')(x)
        x = layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_1')(x)
        x = layers.TimeDistributed(layers.Conv1D(16, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_1')(x)
        x = layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_2')(x)
        x = layers.TimeDistributed(layers.Conv1D(32, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_2')(x)
        x = layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_3')(x)
        x = layers.TimeDistributed(layers.Conv1D(64, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_3')(x)
        
        x = layers.TimeDistributed(layers.Conv1D(128, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_4')(x)
        x = layers.GlobalMaxPooling2D(name='global_max_pooling_2d')(x)
        x = layers.Dropout(rate=0.1, name='dropout')(x)
        x = layers.Dense(64, activation='relu', activity_regularizer=l2(0.001), name='dense')(x)
        o = layers.Dense(N_CLASSES, activation='softmax', name='softmax')(x)

        model = Model(inputs=i.input, outputs=o, name='1d_convolution')
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

        return model
        
        
    def get_conv2D_model(self,N_CLASSES=6, SR=16000, DT=1.0):
        # input shape (n, feat, channel)
        input_shape = (int(SR*DT), 1)
        i = get_melspectrogram_layer(input_shape=input_shape,
                                    n_mels=128,
                                    pad_end=True,
                                    n_fft=512,
                                    win_length=400,
                                    hop_length=160,
                                    sample_rate=SR,
                                    return_decibel=True,
                                    input_data_format='channels_last',
                                    output_data_format='channels_last')
        x = layers.LayerNormalization(axis=2, name='batch_norm')(i.output)
        x = layers.Conv2D(8, kernel_size=(7,7), activation='tanh', padding='same', name='conv2d_tanh')(x)
        x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_1')(x)
        x = layers.Conv2D(16, kernel_size=(5,5), activation='relu', padding='same', name='conv2d_relu_1')(x)
        x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_2')(x)
        x = layers.Conv2D(16, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_2')(x)
        x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_3')(x)
        x = layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_3')(x)
        x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_4')(x)
        x = layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_4')(x)
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dropout(rate=0.2, name='dropout')(x)
        x = layers.Dense(64, activation='relu', activity_regularizer=l2(0.001), name='dense')(x)
        o = layers.Dense(N_CLASSES, activation='softmax', name='softmax')(x)

        model = Model(inputs=i.input, outputs=o, name='2d_convolution')
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

        return model


    def get_recurrent_model(self, N_CLASSES=6, SR=16000, DT=1.0):
        # input shape (n, feat, channel)
        input_shape = (int(SR*DT), 1)
        i = get_melspectrogram_layer(input_shape=input_shape,
                                        n_mels=128,
                                        pad_end=True,
                                        n_fft=512,
                                        win_length=400,
                                        hop_length=160,
                                        sample_rate=SR,
                                        return_decibel=True,
                                        input_data_format='channels_last',
                                        output_data_format='channels_last',
                                        name='rnn')
        x = layers.LayerNormalization(axis=2, name='batch_norm')(i.output)
        x = layers.TimeDistributed(layers.Reshape((-1,)), name='reshape')(x)
        s = layers.TimeDistributed(layers.Dense(64, activation='tanh'),
                            name='td_dense_tanh')(x)
        x = layers.Bidirectional(layers.LSTM(32, return_sequences=True),
                                name='bidirectional_lstm')(s)
        x = layers.concatenate([s, x], axis=2, name='skip_connection')
        x = layers.Dense(64, activation='relu', name='dense_1_relu')(x)
        x = layers.MaxPooling1D(name='max_pool_1d')(x)
        x = layers.Dense(32, activation='relu', name='dense_2_relu')(x)
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dropout(rate=0.2, name='dropout')(x)
        x = layers.Dense(32, activation='relu',
                            activity_regularizer=l2(0.001),
                            name='dense_3_relu')(x)
        o = layers.Dense(N_CLASSES, activation='softmax', name='softmax')(x)

        model = Model(inputs=i.input, outputs=o, name='long_short_term_memory')
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

        return model
    
        
    def get_model(self):
        return self.model
    
    
    def build_rand_feat(self, dataset, class_dist, prob_dist):
        tmp = self.check_data()
        if tmp:
            self.data = (tmp.data[0], tmp.data[1])
            return self
    
        X = []
        y = []
        instrument_list = [
                        'Kamancheh',
                        'Ney',
                        'Santur',
                        'Setar',
                        'Tar',
                        'Ud'
                    ]
        
        X = []
        y = []

        for _ in tqdm(range(90000)):
            rand_class = np.random.choice(class_dist.index, p= prob_dist)
            index = np.random.choice(dataset[dataset['name']==rand_class].index)
            label, wav = dataset.iloc[index]['name'], dataset.iloc[index]['signal']
            rand_index = np.random.randint(0, wav.shape[0]-self.rate)
            sample = wav[rand_index:rand_index+self.rate]
            X.append(sample)
            y.append(instrument_list.index(label))
                
        X,y = np.array(X),np.array(y)
            
        y = to_categorical(y, num_classes=self.n_class)
        X = X.reshape(X.shape[0],X.shape[1],1)
        
        self.data = (X, y)
        
        with open(self.p_path, 'wb') as handle:
                pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    def check_data(self):
        if os.path.isfile(self.p_path):
            print(f"loading existing data for {self.mode} model")
            with open(self.p_path, 'rb') as handle:
                tmp = pickle.load(handle)
                return tmp
        else:
            return None
    
    def fit(self,epoch=10, batch_size=32, validation_split=0.35, verbose=1, monitor='val_acc', mode='max', save_best_only=True, save_weights_only=False, period=1):
        X,y = self.data

        y_flat = np.argmax(y,axis=1)

        class_weights = compute_class_weight('balanced', np.unique(y_flat),y_flat)

        csv_logger = CSVLogger(self.history, append=False)

        checkpoint = ModelCheckpoint(self.model_path, monitor=monitor, verbose=verbose, mode=mode, save_best_only=save_best_only, save_weights_only=save_weights_only, save_freq=period)
        
        self.model.fit(X, y, epochs=epoch, verbose=1, class_weight=class_weights, batch_size=batch_size, shuffle=True, validation_split=validation_split, callbacks=[checkpoint,csv_logger])
        
        self.model.save(self.model_path)
    
    def predict(self,input):
        return self.model.predict(input)
        
    



