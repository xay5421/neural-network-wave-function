import functools
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as utils
from torch import autograd

BASIS = [torch.tensor([1., 0.]), torch.tensor([0., 1.])]
class RNN_wavefunction(nn.Module):

    def __init__(self, system_size, units):
        super(RNN_wavefunction, self).__init__()
        self.N = system_size
        self.input_dim = 2
        self.units = units
        self.nlayer = len(units)
        self.cell = nn.RNNCell
        self.rnn = self.multilayerRNN(units, nonlinearity='tanh')
        self.dense = nn.Linear(units[-1], 2)  # phase & amplitude
        self.outlayer_act = nn.functional.softmax

    def multilayerRNN(self, units, nonlinearity):
        """

        Args:
            units (list): list of meta-cell
            nonlinearity (string): type of non-linear activation function

        Returns:

        """
        return nn.ModuleList([self.cell(self.input_dim, units[0], nonlinearity)]
                             + [self.cell(units[i], units[i + 1], nonlinearity) for i in range(len(units) - 1)])

    def sampling(self, n_sample):
        """
            sampling samples according to p(s)
        Args:
            n_sample ():

        Returns:

        """
        batch = n_sample

        inputs = torch.zeros(batch, self.input_dim, dtype=torch.float)
        hx = [torch.zeros(batch, self.units[li], dtype=torch.float) for li in range(self.nlayer)]

        samples = []
        for i in range(self.N):
            # ------- sigma_i
            hx_ = inputs
            tmp_hx = []

            for li in range(self.nlayer):
                hx_i = hx[li]
                hx_ = self.rnn[li](hx_, hx_i)
                tmp_hx.append(hx_)
            hx = tmp_hx

            rnn_output = tmp_hx[-1]
            o1 = self.dense(rnn_output)
            prob_i = self.outlayer_act(o1, dim=1)

            sample_i = torch.reshape(torch.multinomial(prob_i, num_samples=1),(-1,))
            samples.append(sample_i)
            inputs = torch.nn.functional.one_hot(sample_i, num_classes=self.input_dim).float()

        samples = torch.stack(samples, dim=1)
        # print(samples)
        return samples

    def log_prob(self, samples):
        """
            query the log_prob(s)
        Args:
            samples (tensor):  N x batch_size x dim
                N: system size
                dim: dim of input space
        Returns:

        """
        batch = samples.size(1)
        if self.N != samples.size(0):
            raise ValueError("DimErr: sample=(batch,len,dim)->  len != system size, please check. ")

        inputs = torch.zeros(batch, self.input_dim, dtype=torch.float)
        hx = [torch.zeros(batch, self.units[li], dtype=torch.float) for li in range(self.nlayer)]
        probs = []
        for i in range(self.N):
            #------- sigma_i
            hx_ = inputs
            tmp_hx = []

            for li in range(self.nlayer):
                hx_i = hx[li]
                hx_ = self.rnn[li](hx_, hx_i)
                tmp_hx.append(hx_)
            hx = tmp_hx

            rnn_output = tmp_hx[-1]
            o1 = self.dense(rnn_output)
            prob_i = self.outlayer_act(o1,dim=1)
            probs.append(prob_i)
            # print("input: \n{}\nprob_i = {}".format(inputs, prob_i))
            inputs = samples[i]

        _prob = torch.stack(probs)

        p = torch.einsum('nbi->nb', torch.multiply(_prob, samples))# get list of probability of P(\sigma_{i}|\sigma_{<i})

        self.probs = torch.prod(p, dim=0, keepdim=True) # P(\bm{\sigma})

        self.log_prob = torch.sum(torch.log(p), dim=0, keepdim=True) # logP(\bm{\sigma})

        return self.log_prob


def generate_random_samples(N, batch, basis):
    samples = []
    for i in range(N):
        s = [basis[i] for i in np.random.choice(len(basis), batch)]
        samples.append(torch.stack(s))
    samples = torch.stack(samples)
    return samples


def generate_full_samples(N, basis):
    full_basis = list(itertools.product(range(len(basis)), repeat=N))
    s = torch.tensor(full_basis).T

    batch = []
    for bi in s:
        site_i = []
        for i in bi:
            site_i.append(basis[i])
        batch.append(torch.stack(site_i))
    samples = torch.stack(batch)
    # for i in range(N):
    #     print("\nsite i={}\n{}\n dim={}".format(i, samples[i], samples[i].shape))
    return samples

def generate_sample_test():
    N = 3

    # ---full samples: [N, 2^N, input_dim]
    samples = generate_full_samples(N, basis=BASIS)
    for i in range(N):
        print(samples[i])

    #---random samples: [N, batch, input_dim]
    rsamples = generate_random_samples(N, batch=1, basis=BASIS)
    for i in range(N):
        print(rsamples[i])

def rnn_log_prob_test():
    torch.manual_seed(666)

    units = [10, 20]
    input_dim = 2
    print([(input_dim, units[0])] + [(units[i], units[i + 1]) for i in range(len(units) - 1)])

    N = 3
    batch = 3
    units = [10, 20]
    rnn_state = RNN_wavefunction(system_size=N, units=units)

    samples = generate_random_samples(N, 1, basis=BASIS)
    log_prob = rnn_state.log_prob(samples)

    probs_ = torch.sum(torch.exp(torch.sum(log_prob, dim=0, keepdim=True)))

    # s = rnn_state.parameters()
    # p = [param for param in s]
    # grads = autograd.grad(probs_, rnn_state.parameters())
    print("prob--------", probs_)

def rnn_grad_test():
    torch.manual_seed(666)

    units = [10, 20]
    input_dim = 2
    print([(input_dim, units[0])] + [(units[i], units[i + 1]) for i in range(len(units) - 1)])

    N = 3
    batch = 3
    units = [10, 20]
    rnn_state = RNN_wavefunction(system_size=N, units=units)

    samples = generate_random_samples(N, 1, basis=BASIS)
    log_prob = rnn_state.log_prob(samples)

    probs_ = torch.sum(torch.exp(torch.sum(log_prob, dim=0, keepdim=True)))

    ## autograd
    rnn_state.zero_grad()
    probs_.backward()
    with torch.no_grad():
        for p in rnn_state.parameters():
            gp = p.grad
            op = p

def rnn_sampling_test():
    torch.manual_seed(111)

    units = [10, 20]
    input_dim = 2
    print([(input_dim, units[0])] + [(units[i], units[i + 1]) for i in range(len(units) - 1)])

    N = 3
    batch = 3
    units = [10, 20]
    rnn_state = RNN_wavefunction(system_size=N, units=units)
    params = rnn_state.parameters()
    p = utils.parameters_to_vector(params)
    samples = generate_full_samples(N, basis=BASIS)

    log_prob = rnn_state.log_prob(samples)
    probs_ = torch.exp(torch.sum(log_prob, dim=0, keepdim=True))
    print(probs_.detach().numpy())
    n_sample = 2000
    samples = rnn_state.sampling(n_sample=n_sample)
    s = np.sort([int("".join([str(bi) for bi in i]),2) for i in samples.numpy()])
    print(np.bincount(s)/n_sample)
    print("--------", )

if __name__ == '__main__':
    rnn_log_prob_test()





