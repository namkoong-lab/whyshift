from sklearn.svm import SVC
import numpy as np 
import os
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier
import xgboost as xgb
from .methods.model_family import *


def fetch_model(method_name, input_dim=10, num_classes=2):
    if method_name == 'lr':
        return LogisticRegression(input_dim, num_classes)
    elif method_name == 'mlp':
        return MLPClassifier(input_dim, num_classes)
    elif method_name == 'chi_dro':
        return chi_DRO(input_dim, num_classes)
    elif method_name == 'cvar_dro':
        return cvar_DRO(input_dim, num_classes)
    elif method_name == 'cvar_doro':
        return cvar_DORO(input_dim, num_classes)
    elif method_name == 'chi_doro':
        return chi_DORO(input_dim, num_classes)
    elif method_name == 'group_dro':
        return group_DRO(input_dim, num_classes)
    elif method_name == 'marginal_dro':
        return marginal_DRO(input_dim, num_classes)
    elif method_name in ['FairPostprocess', 'FairPostprocess_exp','FairPostprocess_threshold']:
        return FairPostprocess()
    elif method_name in ['FairInprocess', 'FairInprocess_eo', 'FairInprocess_dp', 'FairInprocess_error_parity']:
        return FairInprocess()
    elif method_name == 'jtt':
        return JTT()
    elif method_name == 'svm2':
        return SVM2()
    elif method_name == 'dwr':
        return DWRPreprocess()
    elif method_name == 'rf':
        return RandomForestClassifier()
    elif method_name == 'gbm':
        return GradientBoostingClassifier()
    elif method_name == 'lightgbm':
        return LGBMClassifier()
    elif method_name == 'svm':
        return SVC(cache_size=1000, max_iter=1000)
    elif method_name in ['xgb', 'subg', 'rwy', 'rwg', 'suby']:
        return xgb.XGBClassifier()
    else:
        raise NotImplementedError
