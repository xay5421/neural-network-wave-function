import functools
import itertools
import warnings

import numpy as np
import scipy.sparse.linalg
import torch
import torch.nn as nn
from openfermion import QubitOperator, get_sparse_operator, MolecularData, get_fermion_operator, jordan_wigner, \
    get_ground_state
from openfermionpyscf import run_pyscf
from torch.func import functional_call, vmap, grad

from nn_wave_function.data_utils import generate_full_samples, generate_random_samples, BASIS_VEC, unique_samples, \
    efficient_estiamte_local_energy

if torch.cuda.is_available():
    torch.set_default_device('cuda')
    print("\n---------------codes have run on gpu.------------------\n")
else:
    warnings.warn("cuda is not available, run on CPU", RuntimeWarning)


class RNN_wavefunction(nn.Module):

    def __init__(self, system_size, units, batch_first=True):
        super(RNN_wavefunction, self).__init__()
        self.N = system_size
        self.input_dim = 2
        self.batch_first = batch_first
        self.units = units
        self.nlayer = len(units)
        self.cell = nn.RNNCell
        self.rnn = self.multilayerRNN(units, nonlinearity='tanh')
        self.dense = nn.Linear(units[-1], 2)
        self.mlp = self.multilayerMLP(units)
        self.dense_phase = nn.Linear(units[-1], 1)
        self.mlp_act_func = nn.functional.tanh
        self.outlayer_amp_act = nn.functional.softmax
        self.outlayer_phs_act = nn.functional.softsign

    def multilayerRNN(self, units, nonlinearity):
        """

        Args:
            units (list): list of meta-cell
            nonlinearity (string): type of non-linear activation function

        Returns:

        """
        if len(units) > 1:
            return nn.ModuleList([self.cell(self.input_dim, units[0], nonlinearity=nonlinearity)]
                                 + [self.cell(units[i], units[i + 1], nonlinearity=nonlinearity) for i in
                                    range(len(units) - 1)])
        else:
            return nn.ModuleList([self.cell(self.input_dim, units[0], nonlinearity=nonlinearity)])

    def multilayerMLP(self, units):
        if len(units) > 1:
            return nn.ModuleList([nn.Linear(self.input_dim*self.N, units[0])]
                                 +[nn.Linear(units[i], units[i+1]) for i in range(len(units) - 1)])
        else:
            return nn.ModuleList([nn.Linear(self.input_dim*self.N, units[0])])

    def sampling(self, n_sample):
        """
            sampling samples according to p(s)
        Args:
            n_sample(int): number of samples to sample

        Returns:
            samples: (batch, N, input_dim)
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
            prob_i = self.outlayer_amp_act(o1, dim=1)

            sample_i = torch.reshape(torch.multinomial(prob_i, num_samples=1), (-1,))
            samples.append(sample_i)
            inputs = torch.nn.functional.one_hot(sample_i, num_classes=self.input_dim).float()

        samples = torch.stack(samples, dim=1)
        # print(samples)
        return samples

    def get_log_prob(self, sample):
        probs = self.forward(sample)
        log_prob = torch.log(probs)  # logP(\bm{\sigma})
        return log_prob

    def forward(self, sample):
        """
            calculate the P(sigma)
        Args:
            sample: (batch, N, input_dim)

        Returns:
            P(sigma)： (batch, )
        """
        if self.batch_first:
            batch = sample.size(0)
            if self.N != sample.size(1):
                raise ValueError("DimErr: sample=(batch,len,dim)->  len != system size, please check. ")
        else:
            batch = sample.size(1)
            if self.N != sample.size(0):
                raise ValueError("DimErr: sample=(batch,len,dim)->  len != system size, please check. ")

        inputs = torch.zeros(batch, self.input_dim, dtype=torch.float)
        hx = [torch.zeros(batch, self.units[li], dtype=torch.float) for li in range(self.nlayer)]
        probs = []
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
            prob_i = self.outlayer_amp_act(o1, dim=1)
            probs.append(prob_i)

            if self.batch_first:
                inputs = sample[:, i, :]
            else:
                inputs = sample[i]
            print("input: \n{}\nprob_i = {}".format(inputs, prob_i))
            # print(inputs)

        if self.batch_first:
            _prob = torch.stack(probs, dim=1)
            t = torch.multiply(_prob, sample)
            p = torch.einsum('bni->bn',
                             t)  # get list of probability of P(\sigma_{i}|\sigma_{<i})

            probs = torch.prod(p, dim=1, keepdim=True)  # P(\bm{\sigma})
            # l = []
            # for i in range(p.size(0)):
            #     s = 1
            #     for j in range(p.size(1)):
            #         s = s*p[i,j]
            #     l.append(s)
            # print(torch.sum(torch.stack(l)))

        else:
            _prob = torch.stack(probs, dim=0)
            t = torch.multiply(_prob, sample)
            p = torch.einsum('nbi->nb', t)  # get list of probability of P(\sigma_{i}|\sigma_{<i})
            probs = torch.prod(p, dim=0, keepdim=True)  # P(\bm{\sigma})

        return probs

    def get_amplitude_phase(self, sample):
        """
            calculate the log_amplitude
        Args:
            sample: (batch, N, input_dim)

        Returns:
            log_amplitude_phase: (batch, N) log P(sigma) + i Phi(sigma)
            log_amplitude: log P(sigma)
            phase: Phi(sigma)
        """
        if self.batch_first:
            batch = sample.size(0)
            if self.N != sample.size(1):
                raise ValueError("DimErr: sample=(batch,len,dim)->  len != system size, please check. ")
        else:
            batch = sample.size(1)
            if self.N != sample.size(0):
                raise ValueError("DimErr: sample=(batch,len,dim)->  len != system size, please check. ")

        inputs = torch.zeros(batch, self.input_dim, dtype=torch.float)
        hx = [torch.zeros(batch, self.units[li], dtype=torch.float) for li in range(self.nlayer)]
        probs = []
        output = None
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
            output = rnn_output
            o1 = self.dense(rnn_output)
            prob_i = self.outlayer_amp_act(o1, dim=1)
            probs.append(prob_i)

            if self.batch_first:
                inputs = sample[:, i, :]
            else:
                inputs = sample[i]
            print("input: \n{}\nprob_i = {}".format(inputs, prob_i))
            # print(inputs)

        output_phase_i = sample.reshape(batch,-1)
        for li in range(self.nlayer):
            input_ = output_phase_i
            output_phase_i = self.mlp_act_func(self.mlp[li](input_))

        phase = torch.pi * (self.outlayer_phs_act(self.dense_phase(output_phase_i)))

        if self.batch_first:
            _prob = torch.stack(probs, dim=1)
            t = torch.multiply(_prob, sample)
            p = torch.einsum('bni->bn',
                             t)  # get list of probability of P(\sigma_{i}|\sigma_{<i})

            sq_probs = torch.sqrt(torch.prod(p, dim=1, keepdim=True))  # P(\bm{\sigma})
            log_probs = torch.log(sq_probs)
            # l = []
            # for i in range(p.size(0)):
            #     s = 1
            #     for j in range(p.size(1)):
            #         s = s*p[i,j]
            #     l.append(s)
            # print(torch.sum(torch.stack(l)))

        else:
            _prob = torch.stack(probs, dim=0)
            t = torch.multiply(_prob, sample)
            p = torch.einsum('nbi->nb', t)  # get list of probability of P(\sigma_{i}|\sigma_{<i})
            sq_probs = torch.sqrt(torch.prod(p, dim=0, keepdim=True))  # P(\bm{\sigma})
            log_probs = torch.log(sq_probs)

        log_amp_phase = torch.complex(log_probs, phase)

        return log_amp_phase, log_probs, phase


def generate_sample_test():
    N = 3

    # ---full samples: batch_first=true: [2^N, N, input_dim] otherwise: [N, 2^N, input_dim]
    samples = generate_full_samples(N)
    for i in range(N):
        print(samples[i])

    # ---random samples: [batch, N, input_dim]
    rsamples = generate_random_samples(N, batch=1)
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

    samples = generate_random_samples(N, batch)
    log_prob = rnn_state.get_log_prob(samples)

    probs_ = torch.sum(torch.exp(torch.sum(log_prob, dim=0, keepdim=True)))

    # s = rnn_state.parameters()
    # p = [param for param in s]
    # grads = autograd.grad(probs_, rnn_state.parameters())
    print("prob--------", probs_)

def rnn_grad_batch_log_prob_test():
    N = 4
    batch = 3
    units = [10, 20]
    batch_first = False
    rnn_state = RNN_wavefunction(system_size=N, units=units, batch_first=batch_first)

    samples = generate_random_samples(N, batch,
                                      batch_first=batch_first)  # default batch_first = True, if batch_first=True, samples = (batch, N, input_dim)
    log_probs = rnn_state.get_log_prob(samples)

    s = rnn_state.parameters()
    p = [param for param in s]

    ##---------- partial derivative of get_amplitude_phase
    amp_grad_list = []
    phase_grad_list = []
    for i in range(samples.size(1)):
        rnn_state.zero_grad()
        _, log_prob, phase = rnn_state.get_amplitude_phase(samples[:, i, :].unsqueeze(1))
        log_prob.backward(retain_graph=True)
        amp_grads = []
        with torch.no_grad():
            for p in rnn_state.parameters():
                gp = p.grad
                op = p
                amp_grads.append(gp)
        amp_grad_list.append(amp_grads)

        log_prob.backward(retain_graph=True)
        phs_grads = []
        with torch.no_grad():
            for p in rnn_state.parameters():
                gp = p.grad
                op = p
                phs_grads.append(gp)
        phase_grad_list.append(phs_grads)

    ##----------way1: partial derivative of get_log_prob
    grad_list = []
    for i in range(samples.size(1)):
        rnn_state.zero_grad()
        ss = rnn_state.get_log_prob(samples[:, i, :].unsqueeze(1))
        ss.backward(retain_graph=True)
        grads = []
        with torch.no_grad():
            for p in rnn_state.parameters():
                gp = p.grad
                op = p
                grads.append(gp)
        grad_list.append(grads)

    ##----------way2: partial derivative of get_log_prob
    grad_list = []
    for i in range(samples.size(1)):
        rnn_state.zero_grad()
        log_probs[:, i].backward(retain_graph=True)
        grads = []
        with torch.no_grad():
            for p in rnn_state.parameters():
                gp = p.grad
                op = p
                grads.append(gp)
        grad_list.append(grads)

    print("end")


def rnn_vmap_grad_test():
    torch.manual_seed(666)

    units = [10, 20]
    input_dim = 2
    print([(input_dim, units[0])] + [(units[i], units[i + 1]) for i in range(len(units) - 1)])

    N = 3
    batch = 2
    units = [6,10]
    rnn_state = RNN_wavefunction(system_size=N, units=units).cuda()
    # batch_sample = generate_random_samples(N, batch, basis=BASIS).reshape(batch, N, -1)

    batch_sample = generate_random_samples(N, batch)
    res = rnn_state(batch_sample)  ## here only accept input = (N, input_dim)

    params = {k: v.detach() for k, v in rnn_state.named_parameters()}

    def compute_loss(par, sample):
        # batch = sample.unsqueeze(0)
        loss = functional_call(rnn_state, par, (sample,))
        return loss

    b = 5
    batch_samples = generate_random_samples(N, batch=b).reshape(b, N, -1)
    single_one = generate_random_samples(N, batch=1).squeeze(1)
    ft_compute_grad = compute_loss(params, single_one)

    grads = vmap(grad(compute_loss), in_dims=(None, 0))(params, batch_samples)
    ## the above method are not suitable for multi-layer RNN, only works for N = 1, batch = n, len(units)=1
    print(grads)


def rnn_grad_test():
    torch.manual_seed(666)

    N = 4
    batch = 1
    units = [10, 20]
    rnn_state = RNN_wavefunction(system_size=N, units=units)

    samples = generate_random_samples(N, batch)
    log_probs = rnn_state.get_log_prob(samples)
    sum_prob = torch.sum(torch.exp(torch.sum(log_probs, dim=0, keepdim=True)))

    ## autograd
    rnn_state.zero_grad()
    sum_prob.backward()
    grads = []
    with torch.no_grad():
        for p in rnn_state.parameters():
            gp = p.grad
            op = p
            grads.append(gp)

    print("grads--------", grads)


def rnn_sampling_test():
    torch.manual_seed(111)

    N = 4
    units = [10, 20]
    batch_first = True
    rnn_state = RNN_wavefunction(system_size=N, units=units, batch_first=batch_first)
    samples = generate_full_samples(N, batch_first=batch_first)

    log_prob = rnn_state.get_log_prob(samples)
    if batch_first:
        probs_ = torch.exp(torch.sum(log_prob, dim=1, keepdim=True))
    else:
        probs_ = torch.exp(torch.sum(log_prob, dim=0, keepdim=True))
    print("target prob dist: \n", probs_.cpu().detach().numpy().T)
    n_sample = 2000
    samples = rnn_state.sampling(n_sample=n_sample)
    s = np.sort([int("".join([str(bi) for bi in i]), 2) for i in samples.cpu().numpy()])
    print("estimated prob dist: \n", np.bincount(s) / n_sample)
    print("--------")

def rnn_eloc_test():
    torch.manual_seed(1231)

    N = 5
    batch = 4
    batch_first = True
    units = [10, 20]
    rnn_state = RNN_wavefunction(system_size=N, units=units)

    s = generate_random_samples(N, batch, basis=BASIS_VEC, batch_first=batch_first)  # (batch, N, input_dim)
    us, us_counts = unique_samples(s, batch_first=batch_first)
    # int_samples, bit_samples = vec_to_string(us, N)
    qo = (QubitOperator('Z0 Y1 X3', 1.7) + QubitOperator('X3 Z4', -3.17)
          + QubitOperator('Y2 Z3', 2.16))

    cc, s_ = efficient_estiamte_local_energy(qo, s, N)
    log_amp_phase_s, log_probs_s, phase_s = rnn_state.get_amplitude_phase(s)
    log_amp_phase_s_, log_probs_s_, phase_s_ = torch.vmap(rnn_state.get_amplitude_phase, in_dims=(0))(s_)
    si = torch.einsum("bm,bm->b", cc, torch.squeeze(log_amp_phase_s_, dim=2))

    print("done")


def rnn_amplitude_phase_test():
    torch.manual_seed(666)

    units = [10, 20]
    input_dim = 2
    print([(input_dim, units[0])] + [(units[i], units[i + 1]) for i in range(len(units) - 1)])

    N = 3
    batch = 4
    units = [10, 20]
    rnn_state = RNN_wavefunction(system_size=N, units=units)

    ##-----vmap batch samples
    samples = torch.stack([generate_full_samples(N) for i in range(3)])
    s = torch.vmap(rnn_state.get_amplitude_phase, in_dims=(0))(samples)

    samples = generate_full_samples(N)
    log_amp_phase, log_probs, phase = rnn_state.get_amplitude_phase(samples)
    p = torch.sum(torch.abs(torch.exp(log_amp_phase)) ** 2)
    print("done")


if __name__ == '__main__':

    rnn_eloc_test()
    rnn_amplitude_phase_test()
    rnn_grad_batch_log_prob_test()
