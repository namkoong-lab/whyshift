from scipy.optimize import brent
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.autograd import grad
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn import linear_model, ensemble, kernel_approximation, svm
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import xgboost as xgb
from fairlearn.reductions import DemographicParity, EqualizedOdds, \
    ErrorRateParity
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data import random_split
from .robust_loss import RobustLoss, group_dro_criterion, cvar_doro_criterion, chi_square_doro_criterion
from .marginal_dro_criterion import LipLoss, opt_model, marginal_dro_criterion
from .model_util import *


TRAIN_FRAC = 0.8

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, num_units=16, nonlin=nn.ReLU(), dropout_ratio=0.1):
        super().__init__()

        self.dense0 = nn.Linear(input_dim, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(dropout_ratio)
        self.dense1 = nn.Linear(num_units, num_units)
        self.output = nn.Linear(num_units, num_classes)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X.float()))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.output(X)
        return X

class Linear(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.dense0 = nn.Linear(input_dim, num_classes)
        
    def forward(self, X, **kwargs):
        return self.dense0(X.float())

class LogisticRegression():
    def __init__(self, input_dim=9, num_classes=2):
        self.model = Linear(input_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def update(self, config):
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.train_epochs = config["train_epochs"]
    
    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        inputs = X.to(self.device)
        self.model = self.model.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.detach().cpu().numpy()

    def score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        self.model = self.model.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total = y.shape[0]
        return correct / total
    
    def f1score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        self.model = self.model.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return f1_score(y.cpu().numpy(), predicted.cpu().numpy(), average='micro')


    def fit(self, X, y, train_ratio=TRAIN_FRAC, device='cpu'):
        self.device = device 

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)

        self.model = self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # criterion = nn.CrossEntropyLoss()
        
        trainset = TensorDataset(X, y)
        test_abs = int(len(trainset) * train_ratio)
        train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])
        trainloader = torch.utils.data.DataLoader(train_subset,batch_size=self.batch_size, shuffle=True, num_workers=8)
        valloader = torch.utils.data.DataLoader(val_subset, batch_size=self.batch_size, shuffle=True, num_workers=8)
        for epoch in range(0,self.train_epochs+1):
            running_loss = 0.0
            epoch_steps = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_steps += 1
                
            # print("[%d] loss: %.3f" % (epoch + 1,running_loss))
            
            
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = self.criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        # print("accuracy", correct / total)

        return correct / total

class MLPClassifier():
    def __init__(self, input_dim=9, num_classes=2):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = MLP(input_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def update(self, config):
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.train_epochs = config["train_epochs"]
        self.model = MLP(self.input_dim, self.num_classes, num_units=config["hidden_size"], dropout_ratio=config["dropout_ratio"])
    
    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        inputs = X.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.detach().cpu().numpy()

    def predict_proba(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        inputs = X.to(self.device)
        outputs = self.model(inputs)
        # print(outputs.shape)
        return outputs.detach().cpu().numpy()


    def score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        self.model = self.model.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total = y.shape[0]
        return correct / total

    def f1score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        self.model = self.model.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return f1_score(y.cpu().numpy(), predicted.cpu().numpy(), average='macro')

    def fit(self, X, y, train_ratio=TRAIN_FRAC, device='cpu'):
        self.device = device 

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)

        self.model = self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # criterion = nn.CrossEntropyLoss()
        
        trainset = TensorDataset(X, y)
        test_abs = int(len(trainset) * train_ratio)
        train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])
        trainloader = torch.utils.data.DataLoader(train_subset,batch_size=self.batch_size, shuffle=True, num_workers=1)
        valloader = torch.utils.data.DataLoader(val_subset, batch_size=self.batch_size, shuffle=True, num_workers=1)
        for epoch in tqdm(range(0,self.train_epochs+1)):
            running_loss = 0.0
            epoch_steps = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_steps += 1
                
            # print("[%d] loss: %.3f" % (epoch + 1,running_loss))
            
            
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # print("accuracy", correct / total)

        return correct / total

    def fit_weight(self, X, y, weights, train_ratio=TRAIN_FRAC, device='cpu'):
        self.device = device 
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)
            weights = torch.tensor(weights).float()

        self.model = self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # criterion = nn.CrossEntropyLoss()
        
        trainset = TensorDataset(X, y, weights)
        test_abs = int(len(trainset) * train_ratio)
        train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])
        trainloader = torch.utils.data.DataLoader(train_subset,batch_size=self.batch_size, shuffle=True, num_workers=1)
        valloader = torch.utils.data.DataLoader(val_subset, batch_size=self.batch_size, shuffle=True, num_workers=1)
        for epoch in tqdm(range(0,self.train_epochs+1)):
            running_loss = 0.0
            epoch_steps = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels, weights_batch = data
                inputs, labels, weights_batch = inputs.to(self.device), labels.to(self.device), weights_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss = torch.dot(loss.reshape(-1), weights_batch.reshape(-1))/loss.shape[0]
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_steps += 1
                
            # print("[%d] loss: %.3f" % (epoch + 1,running_loss))
            
            
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels, weights_batch = data
                inputs, labels, weights_batch = inputs.to(self.device), labels.to(self.device), weights_batch.to(self.device)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # print("accuracy", correct / total)

        return correct / total

#chi-DRO
class chi_DRO():
    def __init__(self, input_dim=9, num_classes=2, size=0.1, reg=0.1, max_iter=100, device='cpu'):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = MLP(input_dim, num_classes)
        self.criterion = RobustLoss(
            geometry='chi-square',
            size=size,
            reg=reg,
            max_iter=max_iter,
            is_regression=False, device=torch.device("cuda:6"))
        self.device = device

    def update(self, config):
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.train_epochs = config["train_epochs"]
        self.model = MLP(self.input_dim, self.num_classes, num_units=config["hidden_size"], dropout_ratio=config["dropout_ratio"])
        self.criterion = RobustLoss(
            geometry='chi-square',
            size=config["size"],
            reg=config["reg"],
            max_iter=500,
            is_regression=False, device=config["device"])

    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        inputs = X.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.detach().cpu().numpy()

    def score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total = y.shape[0]
        return correct / total

    def f1score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        self.model = self.model.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return f1_score(y.cpu().numpy(), predicted.cpu().numpy(), average='macro')

    def fit(self, X, y, train_ratio=TRAIN_FRAC, device='cpu'):
        self.device = device 

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)

        self.model = self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # criterion = nn.CrossEntropyLoss()
        
        trainset = TensorDataset(X, y)
        test_abs = int(len(trainset) * train_ratio)
        train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])
        trainloader = torch.utils.data.DataLoader(train_subset,batch_size=self.batch_size, shuffle=True)
        valloader = torch.utils.data.DataLoader(val_subset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(0,self.train_epochs+1):
            running_loss = 0.0
            epoch_steps = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_steps += 1
                
            # print("[%d] loss: %.3f" % (epoch + 1,running_loss))
            
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # print("accuracy", correct / total)

        return correct / total

#CVaR-DRO
class cvar_DRO():
    def __init__(self, input_dim=9, num_classes=2, size=0.1, reg=0.1, max_iter=100):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = MLP(input_dim, num_classes)
        self.criterion = RobustLoss(
            geometry='cvar',
            size=size,
            reg=reg,
            max_iter=max_iter,
            is_regression=False, device=torch.device("cuda:6"))

    def update(self, config):
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.train_epochs = config["train_epochs"]
        self.model = MLP(self.input_dim, self.num_classes, num_units=config["hidden_size"], dropout_ratio=config["dropout_ratio"])
        self.device = config["device"]
        self.criterion = RobustLoss(
            geometry='cvar',
            size=config["size"],
            reg=config["reg"],
            max_iter=config["max_iter"],
            is_regression=False, device=config["device"])

    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        inputs = X.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.detach().cpu().numpy()

    def score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total = y.shape[0]
        return correct / total

    def f1score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        self.model = self.model.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return f1_score(y.cpu().numpy(), predicted.cpu().numpy(), average='macro')

    def fit(self, X, y, train_ratio=TRAIN_FRAC, device=torch.device("cpu")):
        assert self.device == device
        self.device = device 

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)

        self.model = self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # criterion = nn.CrossEntropyLoss()
        
        trainset = TensorDataset(X, y)
        test_abs = int(len(trainset) * train_ratio)
        train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])
        trainloader = torch.utils.data.DataLoader(train_subset,batch_size=self.batch_size, shuffle=True, num_workers=1)
        valloader = torch.utils.data.DataLoader(val_subset, batch_size=self.batch_size, shuffle=True, num_workers=1)
        # print("Begin fitting..")
        for epoch in range(0,self.train_epochs+1):
            running_loss = 0.0
            epoch_steps = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_steps += 1
                # print(i)
                
            # print("[%d] loss: %.3f" % (epoch + 1,running_loss))
            
            
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # print("accuracy", correct / total)

        return correct / total

#CVaR-DORO
class cvar_DORO():
    def __init__(self, input_dim=9, num_classes=2):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = MLP(input_dim, num_classes)

    def update(self, config):
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.train_epochs = config["train_epochs"]
        self.model = MLP(self.input_dim, self.num_classes, num_units=config["hidden_size"], dropout_ratio=config["dropout_ratio"])
        self.eps = config["eps"]
        self.alpha = config["alpha"]


    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        inputs = X.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.detach().cpu().numpy()

    def score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        self.model = self.model.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total = y.shape[0]
        return correct / total

    def f1score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        self.model = self.model.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return f1_score(y.cpu().numpy(), predicted.cpu().numpy(), average='macro')

    def fit(self, X, y, train_ratio=TRAIN_FRAC, device='cpu'):
        self.device = device 

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)

        self.model = self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        trainset = TensorDataset(X, y)
        test_abs = int(len(trainset) * train_ratio)
        train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])
        trainloader = torch.utils.data.DataLoader(train_subset,batch_size=self.batch_size, shuffle=True, num_workers=1)
        valloader = torch.utils.data.DataLoader(val_subset, batch_size=self.batch_size, shuffle=True, num_workers=1)
        for epoch in range(0,self.train_epochs+1):
            running_loss = 0.0
            epoch_steps = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = cvar_doro_criterion(outputs, labels, self.eps, self.alpha)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_steps += 1
                
            # print("[%d] loss: %.3f" % (epoch + 1,running_loss))
            
            
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # print("accuracy", correct / total)

        return correct / total

#chi-DORO
class chi_DORO():
    def __init__(self, input_dim=9, num_classes=2, eps=0.1, alpha=0.1,lr=1e-3):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = MLP(input_dim, num_classes)
        self.eps = eps 
        self.alpha = alpha 
        self.lr = lr
        self.batch_size=128
        self.train_epochs=100

    def update(self, config):
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.train_epochs = config["train_epochs"]
        self.model = MLP(self.input_dim, self.num_classes, num_units=config["hidden_size"], dropout_ratio=config["dropout_ratio"])
        self.eps = config["eps"]
        self.alpha = config["alpha"]


    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        inputs = X.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.detach().cpu().numpy()

    def score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        self.model = self.model.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total = y.shape[0]
        return correct / total

    def f1score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        self.model = self.model.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return f1_score(y.cpu().numpy(), predicted.cpu().numpy(), average='macro')

    def fit(self, X, y, train_ratio=TRAIN_FRAC, device='cpu'):
        self.device = device 

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)

        self.model = self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        trainset = TensorDataset(X, y)
        test_abs = int(len(trainset) * train_ratio)
        train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])
        trainloader = torch.utils.data.DataLoader(train_subset,batch_size=self.batch_size, shuffle=True, num_workers=1)
        valloader = torch.utils.data.DataLoader(val_subset, batch_size=self.batch_size, shuffle=True, num_workers=1)
        for epoch in range(0,self.train_epochs+1):
            # print(epoch)
            running_loss = 0.0
            epoch_steps = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = chi_square_doro_criterion(outputs, labels, self.eps, self.alpha)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_steps += 1
                
            # print("[%d] loss: %.3f" % (epoch + 1,running_loss))
            
            
        val_loss = 0.0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # print("accuracy", correct / total)

        return correct / total

#GROUP-DRO
class group_DRO():
    def __init__(self, input_dim=9, num_classes=2, n_groups = 2, device='cpu'):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.sensitive_feature = [0]
        self.model = MLP(input_dim, num_classes)
        self.device = device
        self.group_weights = torch.ones(n_groups, device=self.device)
        self.group_weights = self.group_weights / self.group_weights.sum()



    def update(self, config):
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.train_epochs = config["train_epochs"]
        self.sensitive_feature = config["sensitive_features"]
        self.model = MLP(self.input_dim, self.num_classes, num_units=config["hidden_size"], dropout_ratio=config["dropout_ratio"])
        self.group_weights_step_size = torch.Tensor(
            [config["group_weights_step_size"]]).to(self.device)



    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        inputs = X.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.detach().cpu().numpy()

    def score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        self.model = self.model.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total = y.shape[0]
        return correct / total
    
    def f1score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        self.model = self.model.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return f1_score(y.cpu().numpy(), predicted.cpu().numpy(), average='macro')

    def group_info(self, X):
        if isinstance(X, torch.Tensor):
            return torch.Tensor(X.cpu().numpy()[:,self.sensitive_feature]).to(self.device)
        else:
            raise NotImplementedError

    def fit(self, X, y, train_ratio=TRAIN_FRAC, device = "cpu"):
        self.device = device 

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)

        self.model = self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        trainset = TensorDataset(X, y)
        test_abs = int(len(trainset) * train_ratio)
        train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])
        trainloader = torch.utils.data.DataLoader(train_subset,batch_size=self.batch_size, shuffle=True, num_workers=1)
        valloader = torch.utils.data.DataLoader(val_subset, batch_size=self.batch_size, shuffle=True, num_workers=1)
        for epoch in range(0,self.train_epochs+1):
            running_loss = 0.0
            epoch_steps = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                g = self.group_info(inputs)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                group_losses = group_dro_criterion(outputs, labels, g)
                # update group weights
                self.group_weights = self.group_weights * torch.exp(
                    self.group_weights_step_size * group_losses.data)
                self.group_weights = (self.group_weights / (self.group_weights.sum()))
                # update model
                loss = group_losses @ self.group_weights
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_steps += 1
                
            # print("[%d] loss: %.3f" % (epoch + 1,running_loss))
            
            
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # print("accuracy", correct / total)

        return correct / total

# marginal-dro
class marginal_DRO():
    def __init__(self, input_dim=9, num_classes=2, device='cpu'):
        self.input_dim = input_dim
        self.num_classes = num_classes
        # set this to be the shifted covariate
        self.sensitive_feature = [0]
        self.model = MLP(input_dim, num_classes)
        self.device = device
    
    def update(self, config):
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.train_epochs = config["train_epochs"]
        self.model = MLP(self.input_dim, self.num_classes, num_units=config["hidden_size"], dropout_ratio=config["dropout_ratio"])
        self.radius = config["radius"]
        self.p_min = config["p_min"]
        self.nbisect = config["nbisect"]
        self.max_iter = config["max_iter"]
    
    @property
    def rho(self):
        return 1.0 / self.p_min

    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        inputs = X.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.detach().cpu().numpy()

    def score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        self.model = self.model.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total = y.shape[0]
        return correct / total

    def group_info(self, X):
        if isinstance(X, torch.Tensor):
            return torch.Tensor(X.cpu().numpy()[:,self.sensitive_feature]).to(self.device)
        else:
            raise NotImplementedError
    def _init_eta(self, x_in, y_in):
        loss = nn.CrossEntropyLoss()
        sens = self.group_info(x_in)
        criterion = LipLoss(radius=self.radius, x_in=sens,
                       eta=0)
        wrapped_fun = lambda eta: opt_model(self.model, loss, criterion, 0.0,
                                            self.rho, x_in=x_in, y_in=y_in,
                                            lr=self.lr,
                                            niter=self.max_iter)[0][-1]
        
        opt_init = opt_model(self.model, loss, criterion, 0.0, self.rho, x_in,
                             y_in, lr=self.lr, niter=self.max_iter)
        brack_ivt = (min(0, np.nanmin(opt_init[1])), np.nanmax(opt_init[1]))
        bopt = brent(wrapped_fun, brack=brack_ivt, maxiter=self.nbisect,
                     full_output=True)
        eta = bopt[0]
        return eta


    def fit(self, X, y, train_ratio=TRAIN_FRAC, device='cpu'):
        self.device = device 

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)

        self.model = self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        trainset = TensorDataset(X, y)
        test_abs = int(len(trainset) * train_ratio)
        train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])
        trainloader = torch.utils.data.DataLoader(train_subset,batch_size=self.batch_size, shuffle=True, num_workers=8)
        valloader = torch.utils.data.DataLoader(val_subset, batch_size=self.batch_size, shuffle=True, num_workers=8)
        for epoch in range(0,self.train_epochs+1):
            running_loss = 0.0
            epoch_steps = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                g = self.group_info(inputs)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                self.eta_train = self._init_eta(inputs, labels)
                loss = marginal_dro_criterion(outputs, labels, g, self.radius, self.eta_train)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_steps += 1
                
            # print("[%d] loss: %.3f" % (epoch + 1,running_loss))
            
            
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # print("accuracy", correct / total)

        return correct / total

# SVM WITH KERNEL APPROXIMATION
class SVM2():
    def __init__(self):
        self.model = svm.LinearSVC()
        self.preprocessor = kernel_approximation.RBFSampler()
    def update(self, config):
        self.kernel = config['kernel']
        
        if config["kernel"] == 'rbf':
            self.preprocessor = kernel_approximation.RBFSampler(gamma = config["gamma"], n_components = config["n_components"], random_state = 1)
            print(self.preprocessor)
        elif config["kernel"] == 'poly':
            self.preprocessor == kernel_approximation.PolynomialCountSketch(gamma = config["gamma"], n_components = config["n_components"])
        elif config["kernel"] == 'linear':
            pass
        self.model = svm.LinearSVC()
    def fit(self, X, y, device=None):
        if self.kernel != 'linear':
            X = self.preprocessor.fit_transform(X)
        self.model.fit(X, y)

    def predict(self, X):
        if self.kernel != 'linear':
            X = self.preprocessor.fit_transform(X)
        return self.model.predict(X)
    def score(self, X, y):
        predicted = self.predict(X)
        correct = (predicted == y).sum()
        total = y.shape[0]
        return correct / total

# JTT approaches (reuse models), for xgboost models we directly change the sample weight for each learner
class JTT():
    def __init__(self, sensitive_features=[0], base_method='xgb'):
        self.sensitive_features = sensitive_features
        if base_method == 'xgb':
            base_model = xgb.XGBClassifier()
        elif base_method == 'gbm':
            base_model = GradientBoostingClassifier()
        else:
            raise NotImplementedError
        self.model = base_model
    def update(self, config):
        self.model = xgb.XGBClassifier(learning_rate=config["learning_rate"], min_split_loss=config["min_split_loss"],
                    max_depth=config["max_depth"], colsample_bytree=config["colsample_bytree"], colsample_bylevel=config["colsample_bylevel"],
                    max_bin=config["max_bin"], grow_policy=config["grow_policy"])
        self.up_weight = config['lambda']



    def fit(self, X, y, train_ratio=TRAIN_FRAC, device = None):
        train_num = int(train_ratio*X.shape[0])
        trainX = X[:train_num,:]
        trainy = y[:train_num]
        valX = X[train_num:, :]
        valy = y[train_num:]
        # first stage
        self.model.fit(trainX, trainy)
        predicted =  self.model.predict(trainX)
        sample_weight = [self.up_weight if predicted[i] != y[i] else 1 for i in range(len(trainX))]
        # second stage: upweight + retrain
        self.model.fit(trainX, trainy, sample_weight = sample_weight)

        return self.score(valX, valy)
    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        predicted = self.predict(X)
        correct = (predicted == y).sum()
        total = y.shape[0]
        return correct / total

    def f1score(self, X, y):
        predicted = self.predict(X)
        return f1_score(y, predicted, average='macro')

# subsampling approaches
class SubSample():
    def __init__(self, X, y, sensitive_features = [0]):
    # transform to pd.DataFrame
        self.X = X
        self.y = y
        self.sensitive_feature = sensitive_features[0]
        self.df_data = pd.DataFrame(X)
        self.df_data['label'] = y
        pass
    def reload(self, balance_name):
        if balance_name == 'suby':
            if np.mean(self.y) > 1:
                grp_pos = resample(self.df_data[self.df_data['label'] == 1], replace = True, n_samples = len(self.df_data[self.df_data['label'] == 0]))
                new_data = pd.concat([grp_pos, self.df_data[self.df_data['label'] == 0]])
            else:
                grp_pos = resample(self.df_data[self.df_data['label'] == 0], replace = True, n_samples = len(self.df_data[self.df_data['label'] == 1]))
                new_data = pd.concat([grp_pos, self.df_data[self.df_data['label'] == 1]])
            y = new_data['label'].to_numpy()
            del new_data['label']
            X = np.array(new_data)
            sample_weight = np.ones(len(X))

        elif balance_name == 'subg':

            feature_cnt = self.df_data.groupby(self.sensitive_feature).count()['label']
            min_cnt, min_cnt_grp = np.min(feature_cnt), np.argmin(feature_cnt)

            new_data = self.df_data[self.df_data[self.sensitive_feature] == feature_cnt.index[min_cnt_grp]]
            for i, idx in enumerate(feature_cnt.index):
                if i != min_cnt_grp:
                    grp_data = resample(self.df_data[self.df_data[self.sensitive_feature] == idx], replace = True, n_samples = min_cnt)
                    new_data = pd.concat([new_data, grp_data])    
            y = new_data['label'].to_numpy()
            del new_data['label']
            X = np.array(new_data)
            sample_weight = np.ones(len(X))

        
        elif balance_name == 'rwy':
            X = self.X
            y = self.y
            up_weight = np.mean(y) / (1 - np.mean(y))
            sample_weight = [1 if y[i] ==  1 else up_weight for i in range(len(y))]
            sample_weight = (sample_weight / np.sum(sample_weight))*X.shape[0]

        elif balance_name == 'rwg':
            X = self.X
            y = self.y
            feature_cnt = self.df_data.groupby(self.sensitive_feature).count()['label']
            min_cnt, min_cnt_grp = np.min(feature_cnt), np.argmin(feature_cnt)
            sample_weightdict = dict.fromkeys(feature_cnt.index)
            for i, idx in enumerate(feature_cnt.index):
                sample_weightdict[idx] = min_cnt / list(feature_cnt)[i]
            sample_weight = [sample_weightdict[i] for i in self.df_data[self.sensitive_feature]]
            sample_weight = (sample_weight / np.sum(sample_weight))*X.shape[0]

        else:
            raise NotImplementedError
        return X, y, sample_weight

# fairness approaches
class FairPostprocess():
    def __init__(self, sensitive_features=[0], base_method='xgb'):
        self.sensitive_features = sensitive_features
        if base_method == 'xgb':
            base_model = xgb.XGBClassifier()
        elif base_method == 'gbm':
            base_model = GradientBoostingClassifier()
        else:
            raise NotImplementedError
        self.model = ThresholdOptimizer(estimator=base_model, predict_method='predict')

    def update(self, config):
        base_model = xgb.XGBClassifier(learning_rate=config["learning_rate"], min_split_loss=config["min_split_loss"],
                    max_depth=config["max_depth"], colsample_bytree=config["colsample_bytree"], colsample_bylevel=config["colsample_bylevel"],
                    max_bin=config["max_bin"], grow_policy=config["grow_policy"])
        self.kind = config["kind"]
        self.sensitive_features = config["sensitive_features"]
        if config["kind"] == "threshold":
            self.model = ThresholdOptimizer(estimator=base_model, predict_method='predict')
        elif config["kind"] == 'exp':
            self.model = ExponentiatedGradient(estimator=base_model, constraints=DemographicParity(difference_bound=0.02))
        else:
            raise NotImplementedError
    
    def fit(self, X, y, train_ratio=TRAIN_FRAC, device=None):
        train_num = int(train_ratio*X.shape[0])
        trainX = X[:train_num,:]
        trainy = y[:train_num]
        valX = X[train_num:, :]
        valy = y[train_num:]
        self.model.fit(trainX, trainy, sensitive_features=trainX[:,self.sensitive_features])
        return self.score(valX, valy)
    
    def predict(self, X):
        if self.kind == "threshold":
            return self.model.predict(X, sensitive_features=X[:,self.sensitive_features],
                               random_state=0)
        elif self.kind == "exp":
            return self.model.predict(X, random_state=0)
    
    def score(self, X, y):
        predicted = self.predict(X)
        correct = (predicted == y).sum()
        total = y.shape[0]
        return correct / total

    def f1score(self, X, y):
        predicted = self.predict(X)
        return f1_score(y, predicted, average='macro')



class FairInprocess():
    def __init__(self, sensitive_features = [0], base_method = 'xgb'):
        self.sensitive_features = sensitive_features
        if base_method == 'xgb':
            base_model = xgb.XGBClassifier()
        elif base_method == 'gbm':
            base_model = GradientBoostingClassifier()
        else:
            raise NotImplementedError
        self.model = CustomExponentiatedGradient(estimator=base_model,
                                    constraints = EqualizedOdds(),
                                    sensitive_features=self.sensitive_features)
        
        
    def update(self, config):
        base_model = xgb.XGBClassifier(learning_rate=config["learning_rate"], min_split_loss=config["min_split_loss"],
                    max_depth=config["max_depth"], colsample_bytree=config["colsample_bytree"], colsample_bylevel=config["colsample_bylevel"],
                    max_bin=config["max_bin"], grow_policy=config["grow_policy"])
        self.kind = config["kind"]
        self.sensitive_features = config["sensitive_features"]
        # Constraint
        if config["kind"] == "dp":
            constraint = DemographicParity()
        elif config["kind"] == "eo":
            constraint = EqualizedOdds()
        elif config["kind"] == "error_parity":
            constraint = ErrorRateParity()
        else:
            raise NotImplementedError
        # Model
        self.model = CustomExponentiatedGradient(estimator=base_model,
                                            constraints=constraint,
                                            sensitive_features=self.sensitive_features)
    def fit(self, X, y, train_ratio=TRAIN_FRAC, device=None):
        train_num = int(train_ratio*X.shape[0])
        trainX = X[:train_num,:]
        trainy = y[:train_num]
        valX = X[train_num:, :]
        valy = y[train_num:]
        self.model.fit(trainX, trainy)
        return self.score(valX, valy)
    
    def predict(self, X):
        z = self.model.predict_proba(X)
        return np.argmax(z, axis=1)


    def score(self, X, y):
        predicted = self.predict(X)
        correct = (predicted == y).sum()
        total = y.shape[0]
        return correct / total
    
    def f1score(self, X, y):
        predicted = self.predict(X)
        return f1_score(y, predicted, average='macro')

class IRM():
    def __init__(self, input_dim=9, num_classes=2, alpha=0.1, pretrain_epochs=100):
        self.model = MLP(input_dim, num_classes)
        self.alpha=alpha
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.pretrain_epochs = pretrain_epochs
        self.lr = 1e-3
        self.device = torch.device("cuda:7")
        self.train_epochs=100
    
    def update(self, config):
        self.lr = config["lr"]
        self.train_epochs = config["train_epochs"]
        self.pretrain_epochs = config["pretrain_epochs"]
        self.model = MLP(self.input_dim, self.num_classes, num_units=config["hidden_size"], dropout_ratio=config["dropout_ratio"])
        self.alpha = config["alpha"]
    
    def fit(self, X_list, y_list, train_ratio=TRAIN_FRAC, device=torch.device("cuda:7")):
        self.device = device 

        if not isinstance(X_list[0], torch.Tensor):
            X_list = [torch.tensor(x) for x in X_list]
            y_list = [torch.tensor(y) for y in y_list]

        X_train_list = [x[:int(x.shape[0]*train_ratio),:] for x in X_list]
        X_val_list = [x[int(x.shape[0]*train_ratio):,:] for x in X_list]
        y_train_list = [y[:int(y.shape[0]*train_ratio)] for y in y_list]
        y_val_list = [y[int(y.shape[0]*train_ratio):] for y in y_list]
        
        self.model = self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(0,self.train_epochs+1):
            scale = torch.tensor(1.).to(self.device).requires_grad_()
            penalty = 0.0
            loss = 0.0
            for idx in range(len(X_train_list)):
                inputs, labels = X_train_list[idx].to(self.device), y_train_list[idx].to(self.device)

                yhat = self.model(inputs)
                loss += criterion(yhat, labels)
                penalty += grad(criterion(yhat * scale, labels), [scale], create_graph=True)[0].pow(2).mean()
            
            irm_lam = self.alpha if epoch > self.pretrain_epochs else 1.0
            
            loss += irm_lam * penalty
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("[%d] loss: %.3f" % (epoch + 1,running_loss))
            
            
        accs = 0.0
        for idx in range(len(X_val_list)):
            with torch.no_grad():
                inputs, labels = X_val_list[idx].to(self.device), y_val_list[idx].to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                accs += (predicted == labels).sum().item()/inputs.shape[0]

        # print("accuracy", accs / len(X_val_list))

        return accs / len(X_val_list)

class IGA():
    def __init__(self, input_dim=9, num_classes=2, alpha=0.1, pretrain_epochs=100):
        self.model = MLP(input_dim, num_classes)
        self.alpha=alpha
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.pretrain_epochs = pretrain_epochs
        self.lr = 1e-3
        self.device = torch.device("cuda:7")
        self.train_epochs=100
    
    def update(self, config):
        self.lr = config["lr"]
        self.train_epochs = config["train_epochs"]
        self.pretrain_epochs = config["pretrain_epochs"]
        self.model = MLP(self.input_dim, self.num_classes, num_units=config["hidden_size"], dropout_ratio=config["dropout_ratio"])
        self.alpha = config["alpha"]
    
    def fit(self, X_list, y_list, train_ratio=TRAIN_FRAC, device=torch.device("cuda:7")):
        self.device = device 

        if not isinstance(X_list[0], torch.Tensor):
            X_list = [torch.tensor(x) for x in X_list]
            y_list = [torch.tensor(y) for y in y_list]

        X_train_list = [x[:int(x.shape[0]*train_ratio),:] for x in X_list]
        X_val_list = [x[int(x.shape[0]*train_ratio):,:] for x in X_list]
        y_train_list = [y[:int(y.shape[0]*train_ratio)] for y in y_list]
        y_val_list = [y[int(y.shape[0]*train_ratio):] for y in y_list]
        
        self.model = self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(0,self.train_epochs+1):
            scale = torch.tensor(1.).to(self.device).requires_grad_()
            penalty = 0.0
            loss = 0.0
            grad_list = []
            for idx in range(len(X_train_list)):
                inputs, labels = X_train_list[idx].to(device), y_train_list[idx].to(device)

                yhat = self.model(inputs)
                loss += criterion(yhat, labels)
                grad_list.append(grad(criterion(yhat * scale, labels), [scale], create_graph=True)[0])
            
            grads = torch.stack(grad_list)
            
            penalty = torch.var(grads, dim=0).sum()


            irm_lam = self.alpha if epoch > self.pretrain_epochs else 1.0
            
            loss += irm_lam * penalty
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("[%d] loss: %.3f" % (epoch + 1,running_loss))
            
            
        accs = 0.0
        for idx in range(len(X_val_list)):
            with torch.no_grad():
                inputs, labels = X_val_list[idx].to(self.device), y_val_list[idx].to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                accs += (predicted == labels).sum().item()/inputs.shape[0]

        # print("accuracy", accs / len(X_val_list))

        return accs / len(X_val_list)


class DWRPreprocess():
    def __init__(self):
        self.base_model = xgb.XGBClassifier()
        self.device = torch.device("cuda:7")
        self.lmbd_l1 = 0.1
        self.lmbd_l2 = 0.1
        self.weight = None
        self.threshold = 1e-3
        
    def update(self, config):
        self.base_model = xgb.XGBClassifier(learning_rate=config["learning_rate"], min_split_loss=config["min_split_loss"],
                    max_depth=config["max_depth"], colsample_bytree=config["colsample_bytree"], colsample_bylevel=config["colsample_bylevel"],
                    max_bin=config["max_bin"], grow_policy=config["grow_policy"])
        self.device = config["device"]
        self.weight = None
        self.lmbd_l1 = config["l1"]
        self.lmbd_l2 = config["l2"]
        self.threshold = config["threshold"]

    def covfunc(self, x, w=None):
        if w is None:
            n = x.shape[0]
            cov = torch.matmul(x.T, x) / n
            e = torch.mean(x, dim=0).view(-1, 1)
            res = cov - torch.matmul(e, e.T)
        else:
            w = w.view(-1, 1)
            cov = torch.matmul((w * x).T, x)
            e = torch.sum(w * x, dim=0).view(-1, 1)
            res = cov - torch.matmul(e, e.T)
        return res

    def corr(self, x, w=None):
        if w is None:
            n = x.shape[0]
            w = torch.ones(n) / n
        w = w.view(-1, 1)
        covariance = self.covfunc(x, w)
        var = torch.sum(w * (x * x), dim=0).view(-1, 1) - torch.square(
            torch.sum(w * x, dim=0).view(-1, 1)
        )
        index = torch.where(var<=0)[0]
        # if torch.min(var) < 0:
        #     print("error: ", torch.min(var))
        std = torch.sqrt(var)
        res = covariance
        # res = covariance / torch.matmul(std, std.T)
        return res

    def decorrelate_loss(self, x, weight, mask):
        assert weight.shape == (x.shape[0], 1)

        w = weight * weight
        n, p = x.shape
        res = self.corr(x, w / torch.sum(w))
        res = res * mask
        loss = torch.mean(res * res)
        
        balance_loss = loss
        loss_weight_sum = (torch.mean(w) - 1) ** 2
        loss_weight_l2 = torch.mean(w ** 2)
        loss = (
            1000 * loss + self.lmbd_l1 * loss_weight_sum + self.lmbd_l2 * loss_weight_l2
        )
        return loss, balance_loss, loss_weight_sum, loss_weight_l2

    def reweight(self, X):
        if self.weight is None:
            x = torch.from_numpy(X).type(torch.float).to(self.device)

            self.weight = torch.ones(x.shape[0], 1).to(self.device)
            num_step = 2000
            tot = 1e-8
            self.weight.requires_grad = True
            mask = 1 - torch.eye(x.shape[1]).to(self.device)

            optimizer = torch.optim.Adam([self.weight], lr=0.001)
            l_pre = 0
            for i in range(num_step):
                optimizer.zero_grad()
                loss, balance_loss, loss_s, loss_2 = self.decorrelate_loss(
                    x, self.weight, mask
                )
                loss.backward()
                optimizer.step()
                if torch.abs(loss - l_pre) <= tot:
                    break
                l_pre = loss


            self.weight.requires_grad = False
            # print('mean', self.weight.mean())
            self.weight = self.weight * self.weight
            
            self.weight = self.weight / torch.mean(self.weight)
        return self.weight.detach().cpu().numpy()

    def fit(self, X, y, device):
        self.device = device
        # print(X.shape)
        self.weight = self.reweight(X).squeeze()
        
        self.feature_selection_model = linear_model.LogisticRegression()
        self.feature_selection_model.fit(X,y, self.weight)
        feature_coefs = np.abs(self.feature_selection_model.coef_).reshape(-1)
        selected_index = np.where(feature_coefs>self.threshold)[0]
        self.feature_list = selected_index
        # print(self.feature_list)
        self.base_model.fit(X[:, self.feature_list], y, self.weight)
        return self.score(X,y)
        
    def score(self, X, y):
        predicted = self.base_model.predict(X[:, self.feature_list])
        correct = (predicted == y).sum()
        total = y.shape[0]
        return correct / total
    
    def f1score(self, X, y):
        predicted = self.base_model.predict(X[:, self.feature_list])
        return f1_score(y, predicted, average='macro')
    
    def predict(self, X):
        return self.base_model.predict(X[:, self.feature_list])


marginal_DRO_test_config = {
    "method": "marginal_dro",
    "device": torch.device("cuda"),
    "lr": 1e-1,
    "batch_size": 1024,
    "hidden_size": 32,
    "dropout_ratio": 0.1,
    "train_epochs": 500,
    "size": 0.4,
    "max_iter": 100,
    "radius": 1,
    "p_min": 0.1,
    "nbisect": 10
}

group_DRO_test_config = {
    "method": "marginal_dro",
    "device": torch.device("cuda"),
    "group_weights_step_size":0.01,
    "lr": 1e-1,
    "batch_size": 1024,
    "hidden_size": 32,
    "dropout_ratio": 0.1,
    "train_epochs": 50,
    "max_iter": 100,
    "size": 0.4
}
SVM_test_config = {
    "method": "svm",
    "gamma": 1,
    "C":1,
    "kernel": "rbf",
    "n_components":2000,
}


# if __name__ == "__main__":
#     # source_X = np.load('./save_data/source_X_trainincome_CA_FairThresholdOptimizer.npy').astype('float')
#     # source_y = np.load('./save_data/source_y_trainincome_CA_FairThresholdOptimizer.npy').astype('int')

#     # # method = chi_DORO(input_dim=source_X.shape[1])
#     # # method.fit(source_X, source_y)
    
#     # method = IRM(input_dim=source_X.shape[1])
#     # method.fit([source_X, source_X], [source_y, source_y])

#     source_X = np.load('./save_data/source_Xincome_CA_chi_doro.npy').astype('float')
#     source_y = np.load('./save_data/source_yincome_CA_chi_doro.npy').astype('int')
    
#     method = chi_DORO(input_dim=source_X.shape[1])
#     method.fit(source_X, source_y)
#     method.score(source_X, source_y)
#     # print(method.f1score(source_X, source_y))
#     # method = DWRPreprocess()
#     # acc = method.fit(source_X, source_y)
#     # print(acc)

#     # method = group_DRO(input_dim = source_X.shape[1])
#     # method.update(group_DRO_test_config)
#     # acc = method.fit(source_X, source_y)

#     # method = svm.SVC(cache_size = 1000, kernel = 'linear', C = 10, gamma = 2)
#     # method.fit(source_X, source_y)
#     # result = method.predict(source_X)
#     # print(accuracy_score(result, source_y))

#     # source_X = np.load('./save_data/source_X_trainincome_CA_rf.npy').astype('float')
#     # source_y = np.load('./save_data/source_y_trainincome_CA_rf.npy').astype('int')

#     # method2 = RandomForestClassifier()
#     # method2.fit(source_X, source_y)
#     # result = method2.predict(source_X)
#     # print(accuracy_score(result, source_y))
    
#     # method = SVM2()
#     # method.update(SVM_test_config)
#     # method.fit(X = source_X, y = source_y, device = device)
#     # result = method.predict(source_X)
#     # print(accuracy_score(result, source_y))

#     # method = svm.LinearSVC()
#     # from sklearn.kernel_approximation import RBFSampler
#     # rbf_feature = RBFSampler(gamma=1, random_state=1, n_components = 2000)
#     # feature_X = rbf_feature.fit_transform(source_X)
#     # method.fit(X = feature_X, y = source_y)
#     # result = method.predict(feature_X)
#     # print(accuracy_score(result, source_y))

#     # method = chi_DORO(input_dim=source_X.shape[1])
#     # method.fit(source_X, source_y)
    
#     # method = IGA(input_dim=source_X.shape[1])
#     # method.fit([source_X, source_X], [source_y, source_y])