# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 15:21:35 2018

@author: DRR
"""
import pandas as pd
import numpy as np

import time
from datetime import date, timedelta
import numpy as np
def base_share_rand_classifier(n,b):
    num_class = len(b)
    #n: number of persons in day d
    #b: historical base shares (or proportions) for each class
    classification = [] #stores the class given to each person
    cumu_p = np.insert(np.cumsum(b),0,0) #creating cummulative probabilities
    #for each person k, randomly classify person using cummulative probs
    for k in range(0,n):
        #generate random number
        w = np.random.random()
        for j in range(0,num_class):
            #classificy person based on in which interval random number falls
            if (cumu_p[j]<= w) and (w <= cumu_p[j+1]):
                classification.append(j+1)
                break


    return classification, classification.count(1)

