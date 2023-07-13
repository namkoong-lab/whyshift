"""
Solve the dual version of the lipschlitz risk bound. (non-sparse, for now).


Solve - min
|| [ L - (b-b^T)1 ]_+ ||_2 / sqrt(n) + tr(Db)R
s.t. b >= 0
"""

import copy
import time
from collections import Counter

from tqdm import tqdm
import numpy as np
import torch
from scipy.optimize import brent
from scipy.spatial.distance import pdist, squareform
from torch import optim
from torch.autograd import Variable
from torch.nn import Module, Parameter
from torch.nn.functional import binary_cross_entropy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pair_dist(x):
    """ given N x D input, return N x N pairwise euclidean dist matrix"""
    return squareform(pdist(x.cpu()))


def outer_chisq_terms(risk, rho, eta):
    """
    :param risk: the expected eta-discounted risk E[ [l(x)-eta]_+^2]^{1/2}
    :return: the chi-sq bound c_k E[ [l(x)-eta]_+^2]^{1/2} + eta
    """
    return rho * risk + eta


class PerformanceLog:
    def __init__(self):
        self.tempTimers = {}
        self.timerDict = Counter()
        self.counterDict = Counter()
        self.data_n = 0
        self.data_dim = 0

    def countTag(self, tag):
        self.counterDict[tag] += 1

    def startTimer(self, tag):
        self.tempTimers[tag] = time.time()

    def stopTimer(self, tag):
        end_time = time.time()
        start_time = self.tempTimers.pop(tag)
        total_time = end_time - start_time
        self.timerDict[tag] += total_time
        self.counterDict[tag] += 1

    def tostr(self):
        strlist = ['--- data property ---',
                   'n_size:' + str(self.data_n) + ',' + 'd_size:' + str(
                       self.data_dim),
                   '--- total times ---',
                   str(self.timerDict),
                   '--- total counts ---',
                   str(self.counterDict)]
        return '\n'.join(strlist)

    def print(self):
        print(self.tostr())

    def reset(self):
        self.timerDict = Counter()
        self.counterDict = Counter()
        self.tempTimers = {}


global_log = PerformanceLog()


class OracleLoss(Module):
    def __init__(self, group_list):
        super(OracleLoss, self).__init__()
        self.group_list = []
        for subgroup in group_list:
            new_var = Variable(torch.LongTensor(subgroup)).to(device)
            self.group_list.append(new_var)

    def forward(self, losses):
        loss_vals = Variable(torch.zeros(len(self.group_list))).to(device)
        for i, subgroup in enumerate(self.group_list):
            group_loss = torch.index_select(losses, 0, subgroup)
            loss_vals[i] = torch.mean(group_loss)
        return torch.max(loss_vals)

    def project(self):
        pass

    def get_beta(self):
        return None


class CVAR(Module):
    """ CVAR loss """

    def __init__(self, alpha):
        super(CVAR, self).__init__()
        self.alpha = alpha
        self.relu = torch.nn.ReLU()
        self.radius = np.inf

    def forward(self, losses, eta):
        """
        :param losses: vector of losses incurred over each example
        :param eta: threshold for the loss.
        :return: truncated loss at eta
        """
        residual = losses - eta
        trunc_map = torch.mean(self.relu(residual))
        return trunc_map

    def project(self):
        pass

    def get_beta(self):
        return None

    def outer(self, risk, rho, eta):
        return risk / (self.alpha) + eta


class ChiSq(Module):
    """ Chi-sq loss """

    def __init__(self, rho, nsize):
        super(ChiSq, self).__init__()
        self.rho = Variable(torch.FloatTensor(np.array([rho]))).to(device)
        self.sqrtn = np.sqrt(nsize)
        self.relu = torch.nn.ReLU()
        self.radius = np.inf

    def forward(self, losses, eta):
        """
        :param b:
        :return: || [ L - (b_+ - b^T_+)1]_+ \\_2 / sqrt(n) + tr(Db)R
        """
        residual = losses - eta
        trunc_map = torch.sqrt(torch.sum(self.relu(residual) ** 2.0)) / (
            self.sqrtn)
        return trunc_map

    def project(self):
        pass

    def get_beta(self):
        return None

    def outer(self, *args, **kwargs):
        return outer_chisq_terms(*args, **kwargs)


class LipLossZO(Module):
    """
    input params - L, n , R, D'
    forward map - evaluate the fn
    """

    def __init__(self, radius, x_in, b_init):
        super(LipLossZO, self).__init__()
        self.b_var = Parameter(torch.FloatTensor(b_init))
        self.sqrtn = np.sqrt(x_in.shape[0])
        self.radius = radius
        dist_np = pair_dist(x_in)
        self.dists = Variable(torch.FloatTensor(dist_np),
                              requires_grad=False).to(
            device
        )
        self.relu = torch.nn.ReLU()

    def forward(self, losses, eta):
        """
        :param b:
        :return: || [ L - (b_+ - b^T_+)1]_+ \\_2 / sqrt(n) + tr(Db)R / n
        """
        bpos = self.b_var
        transport = torch.sum((bpos - torch.t(bpos)), 1)
        residual = losses.squeeze() - transport - eta
        trunc_map = torch.mean(self.relu(residual))
        rad_pen = self.radius
        penalty_term = torch.sum(bpos * self.dists * rad_pen) / (
                    self.sqrtn ** 2.0)
        return trunc_map + penalty_term

    def project(self):
        self.b_var.data = self.relu(self.b_var).data

    def get_beta(self):
        return self.b_var.data.cpu().numpy().copy()

    def outer(self, *args, **kwargs):
        return outer_chisq_terms(*args, **kwargs)


class LipLoss(Module):
    def __init__(self, radius, x_in, eta: float, b_init=None, k_dual=2,
                 loss_fn=torch.nn.functional.binary_cross_entropy):
        super(LipLoss, self).__init__()
        n = x_in.shape[0]

        # b_init is the initial dual variable value (can be set to zero).
        # By default, we set it to the zero matrix (as in utils.py).
        if b_init is None:
            b_init = np.zeros((n, n), dtype=float)

        self.b_var = Parameter(torch.FloatTensor(b_init))
        self.sqrtn = np.sqrt(n)
        self.radius = radius
        self.eta = eta
        dist_np = pair_dist(x_in)
        self.dists = Variable(torch.FloatTensor(dist_np),
                              requires_grad=False).to(device)
        self.relu = torch.nn.ReLU()
        self.k_dual = k_dual
        self.loss_fn = loss_fn

    def forward(self, losses, eta):
        """
        :param b:
        :return: || [ L - (b_+ - b^T_+)1]_+ \\_2 / sqrt(n) + tr(Db)R / n
        """
        bpos = self.b_var
        transport = torch.sum((bpos - torch.t(bpos)), 1)
        if torch.cuda.is_available():
            transport = transport.to("cuda:0")
            bpos = transport.to('cuda:0')

        residual = losses.squeeze() - transport - eta
        kinv = 1.0 / self.k_dual
        kconst = (self.k_dual - 1.0) ** kinv
        trunc_map = kconst * torch.mean(self.relu(residual) ** self.k_dual) ** kinv
        rad_pen = self.radius ** (self.k_dual - 1.0)
        penalty_term = torch.sum(bpos * self.dists * rad_pen) / (self.sqrtn ** 2.0)
        return trunc_map + penalty_term

    def project(self):
        self.b_var.data = self.relu(self.b_var).data

    def get_beta(self):
        return self.b_var.data.cpu().numpy().copy()

    def outer(self, *args, **kwargs):
        return outer_chisq_terms(*args, **kwargs)


class ConfLipLoss(Module):
    """
    input params - L, n , R, D'
    forward map - evaluate the fn
    """

    def __init__(self, radius, x_in, b_init, delta=0):
        super(ConfLipLoss, self).__init__()
        self.b_var = Parameter(torch.FloatTensor(b_init))
        self.sqrtn = np.sqrt(x_in.shape[0])
        self.radius = radius
        dist_np = pair_dist(x_in)
        self.dists = Variable(torch.FloatTensor(dist_np),
                              requires_grad=False).to(
            device
        )
        self.relu = torch.nn.ReLU()
        self.delta = delta

    def forward(self, losses, eta):
        """
        :param b:
        :return: || [ L - (b_+ - b^T_+)1]_+ \\_2 / sqrt(n) + tr(Db)R / n
        """
        bpos = self.b_var
        transport = torch.sum((bpos - torch.t(bpos)), 1)
        residual = losses.squeeze() - transport - eta
        trunc_map = torch.sqrt(torch.sum(self.relu(residual) ** 2.0)) / (
            self.sqrtn)
        penalty_term = torch.sum(bpos * self.dists * self.radius) / (
                    self.sqrtn ** 2.0)
        confounding_penalty = (
                0.5 * self.delta * torch.sum(torch.abs(bpos)) / (
                    self.sqrtn ** 2.0)
        )
        return trunc_map + penalty_term + confounding_penalty

    def project(self):
        self.b_var.data = self.relu(self.b_var).data

    def get_beta(self):
        return self.b_var.data.cpu().numpy().copy()

    def outer(self, *args, **kwargs):
        return outer_chisq_terms(*args, **kwargs)


class RKHSLoss(Module):
    """ RKHS loss - uses gaussian kernel """

    def __init__(self, radius, x_in, b_init, kern_fn):
        super(RKHSLoss, self).__init__()
        self.sqrtn = np.sqrt(x_in.shape[0])
        self.radius = radius
        dist_np = pair_dist(x_in)
        kmat = kern_fn(dist_np, self.radius)
        self.kmat = Variable(torch.FloatTensor(kmat), requires_grad=False).to(
            device)
        self.relu = torch.nn.ReLU()
        self.b_var = Parameter(torch.FloatTensor(b_init))

    def forward(self, losses, eta):
        """
        :param b:
        :return: || [ L - b]_+ \\_2 / sqrt(n) + sqrt(b^T M b)/n
        """
        residual = losses - self.b_var - eta
        trunc_map = torch.mean(self.relu(residual))
        penalty_term = torch.sqrt(
            self.radius * torch.dot(self.b_var,
                                    torch.mv(self.kmat, self.b_var)) + 1e-20
        ) / (self.sqrtn ** 2.0)
        return trunc_map + penalty_term

    def project(self):
        self.b_var.data = self.b_var.data - torch.mean(self.b_var.data)

    def get_beta(self):
        return self.b_var.data.cpu().numpy().copy()

    def outer(self, *args, **kwargs):
        return outer_chisq_terms(*args, **kwargs)


class LinModel(Module):
    def __init__(self, input_dim, use_bias=False, lamb=0):
        super(LinModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1, bias=use_bias)
        self.lamb = lamb

    def forward(self, x):
        return self.linear(x)

    def reg_loss(self):
        reg_loss = 0
        reg_loss += torch.sum(self.linear.weight ** 2.0)
        return self.lamb * reg_loss


class SimLinModel(Module):
    def __init__(self, input_dim, target_dim, use_bias=False, lamb=0):
        super(SimLinModel, self).__init__()
        self.half_dim = int(input_dim / 2)
        self.linear_half = torch.nn.Linear(self.half_dim, target_dim,
                                           bias=use_bias)
        self.b_var = Parameter(torch.FloatTensor([1.0]))
        self.c_var = Parameter(torch.FloatTensor([0.0]))
        self.lamb = lamb

    def split(self, x):
        return x[:, : self.half_dim], x[:, self.half_dim:]

    def forward(self, x):
        x1, x2 = self.split(x)
        v1 = self.linear_half(x1)
        v2 = self.linear_half(x2)
        dvec = torch.sum((v1 - v2) ** 2.0, dim=1)
        dscal = self.b_var * dvec + self.c_var
        return dscal

    def reg_loss(self):
        reg_loss = torch.sum(self.linear_half.weight ** 2.0)
        return self.lamb * reg_loss


class LogitLoss(Module):
    def __init__(self):
        super(LogitLoss, self).__init__()

    def forward(self, y_pred, y_true):
        yprod = (y_pred.squeeze()) * (y_true.squeeze())
        return torch.log(1.0 + torch.exp(-1 * yprod))


class HingeLoss(Module):
    def __init__(self):
        super(HingeLoss, self).__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, y_pred, y_true):
        return self.relu(1.0 - y_pred * y_true)


class SqLoss(Module):
    def __init__(self):
        super(SqLoss, self).__init__()

    def forward(self, y_pred, y_true):
        return (y_pred - y_true) ** 2.0


class AbsLoss(Module):
    def __init__(self):
        super(AbsLoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.abs(y_pred.squeeze() - y_true.squeeze())


def opt_erm(model, loss, x_in, y_in, niter=1000, lr=0.01):
    global_log.startTimer('erm:opt')
    mcuda = model.to(device)
    lcuda = loss.to(device)
    x = Variable(torch.FloatTensor(x_in), requires_grad=False).to(device)
    y = Variable(
        torch.FloatTensor(y_in.astype(float))[:, None], requires_grad=False
    ).to(device)
    optimizer = optim.SGD(mcuda.parameters(), lr=lr)
    loss_trace = []
    for i in range(niter):
        global_log.startTimer('erm:step')
        optimizer.zero_grad()
        y_pred = mcuda.forward(x)
        per_ex_loss = lcuda.forward(y_pred, y)
        reg_loss = mcuda.reg_loss()
        lossval = torch.mean(per_ex_loss) + reg_loss
        lossval.backward()
        optimizer.step()
        loss_trace.append(lossval.data.cpu().numpy())
        global_log.stopTimer('erm:step')
    model_params = copy.deepcopy(
        mcuda
    )
    loss_values = per_ex_loss.data.cpu().numpy().squeeze()
    global_log.stopTimer('erm:opt')
    return loss_trace, loss_values, model_params


def opt_oracle(model, loss, x_in, y_in, z_in, niter=1000, lr=0.01):
    global_log.startTimer('oracle:opt')
    mcuda = model.to(device)
    lcuda = loss.to(device)
    z_idx = [np.nonzero(z_in == i)[0] for i in np.unique(z_in)]
    loss_agg = OracleLoss(z_idx).to(device)
    x = Variable(torch.FloatTensor(x_in), requires_grad=False).to(device)
    y = Variable(
        torch.FloatTensor(y_in.astype(float))[:, None], requires_grad=False
    ).to(device)
    optimizer = optim.SGD(mcuda.parameters(), lr=lr)
    loss_trace = []
    for i in range(niter):
        global_log.startTimer('oracle:step')
        optimizer.zero_grad()
        y_pred = mcuda.forward(x)
        per_ex_loss = lcuda.forward(y_pred, y)
        reg_loss = mcuda.reg_loss()
        lossval = loss_agg(per_ex_loss) + reg_loss
        lossval.backward()
        optimizer.step()
        global_log.countTag('oracle:backward')
        loss_trace.append(lossval.data.cpu().numpy())
        global_log.stopTimer('oracle:step')
    model_params = copy.deepcopy(
        mcuda
    )
    loss_values = per_ex_loss.data.cpu().numpy().squeeze()
    global_log.stopTimer('oracle:opt')
    return loss_trace, loss_values, model_params


def opt_model(
    model, loss, lip_obj, eta_in, rho, x_in, y_in, lr=0.01, niter=1000
):
    classname=lip_obj.__class__.__name__
    global_log.startTimer('lip:opt'+classname)
    model = model.to(device)
    lip_obj = lip_obj.to(device)
    eta = float(eta_in)
    # x = Variable(torch.FloatTensor(x_in), requires_grad=False).to(device)
    # y = Variable(torch.FloatTensor(y_in), requires_grad=False).to(device)
    x = x_in
    y = y_in
    optimizer = optim.SGD(list(model.parameters()) + list(lip_obj.parameters()), lr=lr)
    loss_trace = []
    lip_obj.project()
    for i in tqdm(range(niter)):
        global_log.startTimer('lip:step'+classname)
        optimizer.zero_grad()
        y_pred = model.forward(x)
        per_ex_loss = loss.forward(y_pred, y)
        # We do not need reg_loss term, since our optimizer
        # optionally includes a weight_decay which applies regularization.
        # reg_loss = model.reg_loss()
        # robust_loss = lip_obj.forward(per_ex_loss, eta) + reg_loss
        robust_loss = lip_obj.forward(per_ex_loss, eta)
        robust_loss.backward()
        optimizer.step()
        lip_obj.project()
        global_log.countTag('lip:backward'+classname)
        loss_trace.append(lip_obj.outer(robust_loss.data.cpu().numpy(), rho, eta))
        global_log.stopTimer('lip:step'+classname)
    model_params = copy.deepcopy(
        model
    )
    b_values = lip_obj.get_beta()
    loss_values = per_ex_loss.data.cpu().numpy().squeeze() - eta
    global_log.stopTimer('lip:opt'+classname)
    return loss_trace, loss_values, model_params, b_values


def opt_model_bisect(
    model,
    loss,
    lip_obj,
    rho,
    x_in,
    y_in,
    rad,
    niter_inner=300,
    nbisect=10,
    lr=0.01,
):
    classname=lip_obj.__class__.__name__
    global_log.startTimer('lip:bisect'+classname)
    lip_obj.radius = rad
    wrapped_fun = lambda eta: opt_model(
        model,
        loss,
        lip_obj,
        eta,
        rho,
        x_in,
        y_in,
        lr=lr,
        niter=niter_inner
    )[0][-1]
    opt_init = opt_model(
        model,
        loss,
        lip_obj,
        0.0,
        rho,
        x_in,
        y_in,
        lr=lr,
        niter=niter_inner
    )
    brack_ivt = (min(0, np.nanmin(opt_init[1])), np.nanmax(opt_init[1]))
    bopt = brent(wrapped_fun, brack=brack_ivt, maxiter=nbisect, full_output=True)
    opt_final = opt_model(
        model,
        loss,
        lip_obj,
        bopt[0],
        rho,
        x_in,
        y_in,
        lr=lr,
        niter=niter_inner
    )
    global_log.stopTimer('lip:bisect'+classname)
    return opt_final[0][-1], opt_final, bopt[0]


def marginal_dro_criterion(
        outputs, targets, sens, radius, eta, b_init = None, k_dual = 2):
    lip_loss = LipLoss(radius=radius, x_in=sens, b_init=b_init, k_dual=k_dual,
                       eta=eta)
    elementwise_loss = binary_cross_entropy(input=outputs,
                                            target=targets,
                                            reduction="none")
    loss = lip_loss(losses=elementwise_loss, eta=eta)
    return loss