# implement DISDE method 
# official code for DISDE could be found at https://github.com/namkoong-lab/disde

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
import random



def plot_calibration(prop_p, prop_q, nbins=20, p_weights=None, q_weights=None, 
                     nanmask_threshold=0.01, name='Prop Score',
                     save_dir='.', balance=False):
    
    fig,ax=plt.subplots(1,3,figsize=(10,4))
    for i in range(3):
        ax[i].set_box_aspect(1)
        ax[i].set_xlim(0,1)
    fig.suptitle("Calibration: {}".format(name), fontsize="x-large")

    if p_weights is None: p_weights = np.ones_like(prop_p)
    if q_weights is None: q_weights = np.ones_like(prop_q)

    p_sample_weights = p_weights.copy()
    q_sample_weights = q_weights.copy()
    if balance:
        p_sample_weights = p_sample_weights / p_sample_weights.sum()
        q_sample_weights = q_sample_weights / q_sample_weights.sum()

    conf_scores, bin_edges = np.histogram(np.concatenate([1-prop_p, prop_q]),bins=nbins, density=True, 
                                          weights=np.concatenate([p_sample_weights, 
                                                                 q_sample_weights]),
                                         range=(0,1))
    bin_mids = (bin_edges[1:]+bin_edges[:-1])/2

    nanmask = np.where(conf_scores < nanmask_threshold, np.nan, 1)
    # print(bin_mids)
    # print(nanmask * conf_scores / (conf_scores + conf_scores[::-1]))
    ax[0].plot(bin_mids, nanmask * conf_scores / (conf_scores + conf_scores[::-1]), color='green')
    ax[0].set_ylim(0,1)
    ax[0].set_ylabel('Proportion correct')
    ax[0].set_xlabel('Predicted probability')
    ax[0].set_title('Prop calibration: combined')

    conf_scores, bin_edges = np.histogram(np.concatenate([prop_p]),bins=nbins, weights=p_sample_weights,
                                         range=(0,1))
    bin_mids = (bin_edges[1:]+bin_edges[:-1])/2
    nanmask = np.where(conf_scores < nanmask_threshold, np.nan, 1)

    ax[1].plot(bin_mids, conf_scores, color='green')
    ax[1].set_title('Density: P')
    ax[1].set_xlabel('Predicted probability of Q')
    ax[1].set_ylim(bottom=0)

    conf_scores, bin_edges = np.histogram(np.concatenate([prop_q]),bins=nbins, weights=q_sample_weights,
                                         range=(0,1))
    bin_mids = (bin_edges[1:]+bin_edges[:-1])/2
    nanmask = np.where(conf_scores < nanmask_threshold, np.nan, 1)

    ax[2].plot(bin_mids, conf_scores, color='green')
    ax[2].set_title('Density: Q')
    ax[2].set_xlabel('Predicted probability of Q')
    ax[2].set_ylim(bottom=0)

    fig.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir)

def degradation_decomp(source_X, source_y, other_X_raw, other_y_raw, best_method, data_sum=20000, K=8, domain_classifier=None, draw_calibration=False, save_calibration_png='calibration.png'):
    perm1 = np.random.permutation(other_X_raw.shape[0])
    other_X = other_X_raw[perm1[:data_sum],:]
    other_y = other_y_raw[perm1[:data_sum]]

    piA = np.zeros(source_X.shape[0])
    piB = np.zeros(other_X.shape[0])
    permA = np.random.permutation(piA.shape[0])
    permB = np.random.permutation(piB.shape[0])

    kf = KFold(n_splits=K, shuffle=False)
    A_train_index_list = []    
    A_test_index_list = []    
    B_train_index_list = []    
    B_test_index_list = []    
    for i, (train_index, test_index) in enumerate(kf.split(source_X)):
        A_train_index_list.append(train_index)
        A_test_index_list.append(test_index)
    for i, (train_index, test_index) in enumerate(kf.split(other_X)):
        B_train_index_list.append(train_index)
        B_test_index_list.append(test_index)

    for i in range(K):
        trainX = np.concatenate([source_X[permA[A_train_index_list[i]]],other_X[permB[B_train_index_list[i]]]], axis=0)
        trainT = np.zeros(trainX.shape[0])
        trainT[len(A_train_index_list[i]):] = 1.0
        
        if domain_classifier is None:
            model = XGBClassifier(random_state=0).fit(trainX, trainT)
        else:
            model = domain_classifier.fit(trainX, trainT)
    
        piA[permA[A_test_index_list[i]]] = model.predict_proba(source_X[permA[A_test_index_list[i]]])[:,1]
        piB[permB[B_test_index_list[i]]] = model.predict_proba(other_X[permB[B_test_index_list[i]]])[:,1]

    if draw_calibration:
        plot_calibration(piA, piB, save_dir=save_calibration_png)

    alpha = (other_X.shape[0])/ (source_X.shape[0]+other_X.shape[0])
    wA = piA / ((1-alpha)*piA + alpha * (1-piA))
    wB = (1-piB) / ((1-alpha)*piB + alpha * (1-piB))
    accuracyA = best_method.score(source_X, source_y)
    accuracyB = best_method.score(other_X, other_y)
    wA = wA / np.sum(wA)
    wB = wB / np.sum(wB)
    predA = (best_method.predict(source_X) == source_y)
    predB = (best_method.predict(other_X) == other_y)
    sx_A = np.dot(wA, predA)
    sx_B = np.dot(wB, predB)
    return accuracyA, accuracyB, sx_A, sx_B
