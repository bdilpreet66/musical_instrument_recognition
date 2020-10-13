# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 13:48:22 2020

@author: dilpreet
"""

from models.model import Network
from preprocessing.preprocessing import get_data

if __name__ == "__main__":
    dataset,class_dist = get_data()
    prob_dist = class_dist / class_dist.sum()
    
    model1 = Network(dataset, class_dist, prob_dist, mode='cnn1d')
    
    model1.fit()
    
    model2 = Network(dataset, class_dist, prob_dist, mode='cnn2d')
    
    model2.fit()
    
    model3 = Network(dataset, class_dist, prob_dist, mode='rnn')
    
    model3.fit()

