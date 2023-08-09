import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
import sklearn 
import random
import graphviz 
import xgboost as xgb 
from .dataset import get_data


def shared_reweight(source_X, other_X, K=8):
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
        model = xgb.XGBClassifier(random_state=0)
        model.fit(trainX,trainT)
        piA[permA[A_test_index_list[i]]] = model.predict_proba(source_X[permA[A_test_index_list[i]]])[:,1]
        piB[permB[B_test_index_list[i]]] = model.predict_proba(other_X[permB[B_test_index_list[i]]])[:,1]
    
    alpha = (other_X.shape[0])/ (source_X.shape[0]+other_X.shape[0])
    wA = piA / ((1-alpha)*piA + alpha * (1-piA))
    wB = (1-piB) / ((1-alpha)*piB + alpha * (1-piB))
    wA = wA * 10000 / wA.sum()
    wB = wB * 10000 / wB.sum()
    
    new_X = np.concatenate([source_X, other_X], axis=0)
    new_weights = np.concatenate([wA,wB])
    new_weights /= np.sum(new_weights)

    return wA, wB, new_X, new_weights


def risk_region(method, source_model, target_model, task, source_state, target_state, root_dir='./dataset/acs/', need_preprocess=False, year = 2018):
    num_samples = 10000
    test_results = {}
    test_X = {}
    test_y = {}
    
    sample_sizes = [20000, 10000]
    for idx, state in enumerate([source_state, target_state]):
        source_X_raw, source_y_raw, feature_names = get_data(task, state, need_preprocess, root_dir, year)

        if source_X_raw.shape[0]>=2*num_samples:
            perm2 = np.random.permutation(source_X_raw.shape[0])[:sample_sizes[idx]]
            test_X[state] = source_X_raw[perm2,:]
            test_y[state] = source_y_raw[perm2]
        else:
            perm2 = np.random.permutation(source_X_raw.shape[0])[:sample_sizes[idx]]
            test_X[state] = source_X_raw[perm2,:]
            test_y[state] = source_y_raw[perm2]
            

    wA, wB, new_X, new_weights = shared_reweight(test_X[source_state], test_X[target_state])
    
    if not method == 'mlp':
        source_model.fit(test_X[source_state], test_y[source_state], sample_weight = wA)
        target_model.fit(test_X[target_state], test_y[target_state], sample_weight = wB)    
    else:
        print(wA.shape, wB.shape)
        source_model.fit_weight(test_X[source_state], test_y[source_state], wA.reshape(-1))
        target_model.fit_weight(test_X[target_state], test_y[target_state],wB.reshape(-1))    

    source_acc = source_model.score(test_X[source_state], test_y[source_state])
    target_acc = source_model.score(test_X[target_state], test_y[target_state])

    print("source model state %s acc is %.4f" % (source_state, source_acc))
    print("source model state %s acc is %.4f" % (target_state, target_acc))   

    source_acc2 = target_model.score(test_X[source_state], test_y[source_state])
    target_acc2 = target_model.score(test_X[target_state], test_y[target_state])
    print("tgt model state %s acc is %.4f" % (source_state, source_acc2))
    print("tgt model state %s acc is %.4f" % (target_state, target_acc2))   



    eps = 1e-5
    new_X = np.concatenate([test_X[source_state], test_X[target_state]], axis = 0)
    proba_P2Q = np.clip(source_model.predict_proba(new_X)[:, -1], eps, 1 - eps)
    proba_Q2P = np.clip(target_model.predict_proba(new_X)[:, -1], eps, 1 - eps)
    new_Y = np.abs(proba_P2Q - proba_Q2P)

    plt.figure(figsize=(50,50), dpi=200)
    region_model = DecisionTreeRegressor(max_depth=6,min_samples_leaf=100,min_samples_split=200,min_weight_fraction_leaf=0.05, ccp_alpha=0.0001).fit(new_X, new_Y, sample_weight=new_weights)
    sklearn.tree.plot_tree(region_model, filled=True, feature_names = feature_names)
    plt.savefig(f'{method}_{task}_{source_state}to{target_state}_region.pdf')


  

