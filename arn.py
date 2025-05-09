import functools
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as utils
from torch import autograd

BASIS = [torch.tensor([1., 0.]), torch.tensor([0., 1.])]

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim = 1)
        )

    def forward(self, x):
        return self.mlp(x)

class ARN_wavefunction(nn.Module):

    def __init__(self, system_size, hidden_dim):
        super(ARN_wavefunction, self).__init__()
        self.N = system_size
        self.input_dim = 2
        self.subnets = nn.ModuleList()
        for i in range(self.N):
            input_dim = 2 * i
            self.subnets.append(MLP(input_dim, hidden_dim, 2))

    def sampling(self, n_sample):
        """
            sampling samples according to p(s)
        Args:
            n_sample ():

        Returns:

        """
        batch = n_sample

        inputs = torch.empty(batch, 0)
        samples = []

        for i in range(self.N):
            prob_i = self.subnets[i](inputs)
            sample_i = torch.reshape(torch.multinomial(prob_i, num_samples=1),(-1,))
            samples.append(sample_i)
            inputs = torch.concat([inputs, torch.nn.functional.one_hot(sample_i, num_classes=self.input_dim).float()], dim = 1)

        samples = torch.stack(samples, dim=1)
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
            raise ValueError("DimErr: sample=(len,batch,dim)->  len != system size, please check. ")
        
        inputs = torch.empty(batch, 0)
        probs = []

        for i in range(self.N):
            prob_i = self.subnets[i](inputs)
            probs.append(prob_i)
            inputs = torch.concat([inputs, samples[i]], dim = 1)

        _prob = torch.stack(probs)
        p = torch.einsum('nbi->nb', torch.multiply(_prob, samples)) # get list of probability of P(\sigma_{i}|\sigma_{<i})
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
    print(samples)
    # for i in range(N):
    #     print("\nsite i={}\n{}\n dim={}".format(i, samples[i], samples[i].shape))
    return samples

def general_sample_test():
    N = 3

    samples = generate_full_samples(N, basis = BASIS)

    for i in range(N):
        print(samples[i])
    
    rsamples = generate_random_samples(N, batch = 1, basis = BASIS)
    for i in range(N):
        print(rsamples[i])

def arn_log_prob_test():
    
    N = 10
    batch = 1024
    hidden_dim = 50
    arn_state = ARN_wavefunction(system_size = N, hidden_dim = hidden_dim)

    samples = generate_random_samples(N, batch, basis=BASIS)
    log_prob = arn_state.log_prob(samples)

    print(log_prob)

    probs_ = torch.sum(torch.exp(torch.sum(log_prob, dim=0, keepdim=True)))

    print("prob--------", probs_)

def arn_sampling_test():
    N = 10
    batch = 10
    hidden_dim = 50
    arn_state = ARN_wavefunction(system_size = N, hidden_dim = hidden_dim)

    params = arn_state.parameters()
    p = utils.parameters_to_vector(params)
    samples = generate_full_samples(N, basis=BASIS)

    log_prob = arn_state.log_prob(samples)
    probs_ = torch.exp(torch.sum(log_prob, dim=0, keepdim=True))
    print(probs_.detach().numpy())
    print("sum = ",probs_.sum())
    n_sample = 20000
    samples = arn_state.sampling(n_sample=n_sample)
    s = np.sort([int("".join([str(bi) for bi in i]),2) for i in samples.numpy()])
    print(np.bincount(s)/n_sample)
    print("--------", )


if __name__ == '__main__':
    # general_sample_test()
    # arn_log_prob_test()
    arn_sampling_test()