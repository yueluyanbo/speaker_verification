#!/usr/bin/python
#-*- coding: utf-8 -*-

import os
import glob
import sys
import time
from sklearn import metrics
import numpy
import pdb

def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):
    
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    fpr = fpr*100
    fnr = fnr*100

    print("******************")
    print("fpr:", fpr)
    print("fnr:", fnr)
    print("******************")
    
    tunedThreshold = [];
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);
    
    for tfa in target_fa:
        idx = numpy.nanargmin(numpy.absolute((tfa - fpr))) # numpy.where(fpr<=tfa)[0][-1]
        print("******************")
        print("tfa: ", tfa)
        print("numpy.absolute((tfa - fpr)): ", numpy.absolute((tfa - fpr)))
        print("idx = numpy.nanargmin(numpy.absolute((tfa - fpr))): ", idx)
        print("[thresholds[idx], fpr[idx], fnr[idx]: ", thresholds[idx], fpr[idx], fnr[idx])
        print("******************")
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);
    
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer  = max(fpr[idxE],fnr[idxE])
    
    return (tunedThreshold, eer, fpr, fnr);
