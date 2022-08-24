from sklearn.metrics import recall_score
from functools import partial
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

tnr_score = partial(recall_score, pos_label=0)  # specificity
tpr_score = partial(recall_score, pos_label=1)  # sensitivity

def hma_score(y_true, y_pred):
    """Returns harmonic mean(HMA) of sensitivity(tpr) and specificity(tnr)
    if tpr & tnr are both zero then returns nan"""
    tnr = tnr_score(y_true, y_pred)
    tpr = tpr_score(y_true, y_pred)
    return (2 * tnr * tpr) / (tnr + tpr)

def hma_curve(y_true, y_prob):
    thresholds = np.linspace(0,1,101).round(2) # thresholds with 0.01 step
    hma, tpr, tnr = [], [], []
    for thr in thresholds:
        y_pred = np.zeros(y_prob.shape) # initialize y_pred to zero
        y_pred[y_prob > thr] = 1 # apply threshold
        hma.append(hma_score(y_true, y_pred))
        tpr.append(tpr_score(y_true, y_pred))
        tnr.append(tnr_score(y_true, y_pred))
    # generate hma_df
    hma_df = pd.DataFrame({"thresh":thresholds, "hma":hma, 
                           "tpr":tpr, "tnr":tnr}).round(2)
    return hma_df

def save_hma_curve(hma_df, save_dir, caption):
    plt.figure()
    plt.plot(hma_df.thresh, hma_df.hma, label="hma")
    plt.plot(hma_df.thresh, hma_df.tpr, label="tpr")
    plt.plot(hma_df.thresh, hma_df.tnr, label="tnr")
    plt.title('HMA curve')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.grid()
    plt.legend(loc='best')
    save_path = os.path.join(save_dir, f"{caption}_hma.png")
    plt.savefig(save_path)

def get_classification_scores(y_true, y_pred):
    """To compute specificity(tnr), sensitivity(tpr) and hma (harmonic mean of both)

    Args:
        y_true (1d-list): list of true labels
        y_pred (1d-list): list of predicted labels

    Returns:
        (tnr, tpr, hma)(list): list of scores
    """
    c_scores = (tnr_score(y_true, y_pred), tpr_score(y_true, y_pred), 
            hma_score(y_true, y_pred))
    return np.round(c_scores, 2)