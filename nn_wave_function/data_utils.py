import itertools
import json
import warnings

import numpy as np
import torch
from openfermion import QubitOperator, get_sparse_operator, get_ground_state
from torch.func import vmap
from torch.backends import opt_einsum

from nn_wave_function.utils import project_path, run_time

BASIS_VEC = [torch.tensor([1., 0.]), torch.tensor([0., 1.])]

PAULI_COEFFS = {"X 0": (1., 1), "X 1": (1., 0), "Y 0": (1.j, 1), "Y 1": (-1.j, 0), "Z 0": (1., 0), "Z 1": (-1., 1)}

PAULI_OPS = {'Z': torch.tensor([[1., -1.], [0., 1.]]), 'X': torch.tensor([[1., 1.], [1., 0.]]),
             'Y': torch.tensor([[-1.j, 1.j], [1., 0.]]), 'I': torch.tensor([[1., 1.], [0., 1.]])}

PAULI_OPS_DAG = {'Z': torch.tensor([[1., 0.], [-1., 1.]]), 'X': torch.tensor([[1., 1.], [1., 0.]]),
                 'Y': torch.tensor([[1.j, 1.], [-1.j, 0.]]), 'I': torch.tensor([[1., 0.], [1., 1.]])}

def unique_samples(samples, batch_first=True):
    """
        sample:
            batch_first=True:  (batch_size, N, dim)
            batch_first=False: (N, batch_size, dim)
    """
    if batch_first:
        uni_s = torch.unique(samples, sorted=True, dim=0, return_counts=True)

        # uni_s_ = torch.unique(samples, sorted=False, dim=0, return_counts=True)
        # for i in range(uni_s_[0].size(0)):
        #     print(uni_s_[0][i])
    else:
        uni_s = torch.unique(samples, sorted=True, dim=1, return_counts=True)

        # uni_s_ = torch.unique(samples, sorted=False, dim=1, return_counts=True)
        # for i in range(samples.size(1)):
        #     print(samples[:, i, :])
        # for i in range(uni_s_[0].size(0)):
        #     print(uni_s_[0][i])
    return uni_s


def generate_random_samples(N, batch, basis=BASIS_VEC, batch_first=True):
    # samples = []
    # for i in range(N):
    #     s = [basis[i] for i in np.random.choice(len(basis), batch)]
    #     samples.append(torch.stack(s))
    # samples = torch.stack(samples)
    s = torch.randint(0, 2, (N, batch))
    samples = torch.nn.functional.one_hot(s).view(-1, batch, 2).float()
    if batch_first:
        samples = torch.einsum('nbi->bni', samples)
    return samples


def generate_full_samples(N, basis=BASIS_VEC, batch_first=True):
    full_basis = list(itertools.product(range(len(basis)), repeat=N))
    s = torch.tensor(full_basis).T

    batch = []
    for bi in s:
        site_i = []
        for i in bi:
            site_i.append(basis[i])
        batch.append(torch.stack(site_i))

    if batch_first:
        samples = torch.stack(batch, dim=1)
        for i in range(N):
            print("\nsite i={}\n{}\n dim={}".format(i, samples[:, i, :], samples[:, i, :].shape))
        return samples
    else:
        samples = torch.stack(batch, dim=0)
        for i in range(N):
            print("\nsite i={}\n{}\n dim={}".format(i, samples[i], samples[i].shape))
        return samples


def save_data(path, data, data_type, mode):
    if data_type == 'json':
        with open(project_path + path, mode, encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4)
    elif data_type == 'txt':
        with open(project_path + path, mode, encoding='utf-8') as txt_file:
            txt_file.write(data)

# @run_time
def estimate_local_energy(hamiltonian, states):
    res = []
    for i in range(states.size(0)):
        trans = dict()
        for hi, coeff in hamiltonian.terms.items():
            st = [int(torch.argmax(j)) for j in states[i]]
            tmp_coeff = coeff
            for pos, pauli in hi:
                c, trans_bit = PAULI_COEFFS["{} {}".format(pauli, st[pos])]
                st[pos] = trans_bit
                tmp_coeff *= c
            st = "".join([str(k) for k in st])
            if st in trans:
                trans[st] += tmp_coeff
            else:
                trans[st] = tmp_coeff
        res.append(trans)
    return res

# @run_time
def efficient_estiamte_local_energy(hamiltonian, states, n, is_dag=True):
    if is_dag:
        ##---given s, estimate <s|H\dag
        ops, coeff_h = generate_op_list(hamiltonian, n)
        res = torch.einsum("mnij,bni->bmnj", ops, torch.complex(states, torch.zeros_like(states,device=ops.device)))
        c = res[:, :, :, 0]
        ss = torch.real(res[:, :, :, 1])
        coeff_p = torch.prod(c, dim=2)
        c_ = torch.einsum("ij,j->ij", coeff_p, coeff_h)
    else:
        ##---given s, estimate H|s>
        ops, coeff_h = generate_op_list(hamiltonian, n)
        res = torch.einsum("mnij,bnj->bmni", ops, torch.complex(states, torch.zeros_like(states,device=ops.device)))
        c = res[:, :, :, 0]
        ss = torch.real(res[:, :, :, 1])
        coeff_p = torch.prod(c, dim=2)
        c_ = torch.einsum("ij,j->ij", coeff_p, coeff_h)
    return c_, torch.nn.functional.one_hot(ss.to(torch.long),num_classes=2).to(torch.float)

def unique_test():
    N = 2
    batch = 15
    batch_first = False
    samples = generate_random_samples(N, batch, batch_first=batch_first)
    unique_samples(samples, batch_first=batch_first)

def onehot_to_int(N, vec):
    mask = 2 ** torch.arange(N - 1, -1, -1).to(vec.device, vec.dtype)
    return torch.sum(mask * vec, -1)

def vec_to_string(samples, N, batch_first=True):
    indx = torch.argmax(samples, dim=2)
    if batch_first == False:
        indx = indx.T
    batch_int = vmap(onehot_to_int, in_dims=(None, 0))(N, indx)
    return batch_int, indx

def generate_op_list(hamiltonian, n, is_dag=True):
    if torch.cuda.is_available():
        torch.set_default_device('cuda')
        print("\n---------------codes have run on gpu.------------------\n")
    else:
        warnings.warn("cuda is not available, run on CPU", RuntimeWarning)

    op_list = []
    if is_dag:
        f = PAULI_OPS_DAG
    else:
        f = PAULI_OPS

    coeffs = torch.zeros(len(hamiltonian.terms))
    indx = 0
    for hi, coeff in hamiltonian.terms.items():
        op_ = [f['I'] for _ in range(n)]
        for pos, pauli in hi:
            op_[pos] = f[pauli]
        op_list.append(torch.stack(op_))
        coeffs[indx] = coeff
        indx += 1
        print(hi, coeff)
    ops = torch.stack(op_list).to(coeffs.device)  ##(n_Ham, N, 2, 2)
    return ops, torch.complex(coeffs, torch.zeros_like(coeffs))


def estimat_local_energy_test():

    torch.manual_seed(1231)
    N = 5
    batch = 4
    batch_first = True

    s = generate_random_samples(N, batch, basis=BASIS_VEC, batch_first=batch_first)  # (batch, N, input_dim)
    us, us_counts = unique_samples(s, batch_first=batch_first)
    # int_samples, bit_samples = vec_to_string(us, N)
    qo = (QubitOperator('Z0 Y1 X3', 1.7) + QubitOperator('X3 Z4', -3.17)
          + QubitOperator('Y2 Z3', 2.16))

    r = estimate_local_energy(qo, s)

def efficient_estiamte_local_energy_test():

    torch.manual_seed(1231)

    N = 5
    batch = 4
    batch_first = True

    s = generate_random_samples(N, batch, basis=BASIS_VEC, batch_first=batch_first)  # (batch, N, input_dim)
    us, us_counts = unique_samples(s, batch_first=batch_first)
    # int_samples, bit_samples = vec_to_string(us, N)
    qo = (QubitOperator('Z0 Y1 X3', 1.7) + QubitOperator('X3 Z4', -3.17)
          + QubitOperator('Y2 Z3', 2.16))

    r = efficient_estiamte_local_energy(qo, s, N)

    print("done!")

def eloc_test():

    torch.manual_seed(1231)
    N = 5
    batch = 4
    batch_first = True

    s = generate_random_samples(N, batch, basis=BASIS_VEC, batch_first=batch_first)  # (batch, N, input_dim)
    us, us_counts = unique_samples(s, batch_first=batch_first)
    # int_samples, bit_samples = vec_to_string(us, N)
    qo = (QubitOperator('Z0 Y1 X3', 1.7) + QubitOperator('X3 Z4', -3.17)
          + QubitOperator('Y2 Z3', 2.16))

    r0 = estimate_local_energy(qo, s)
    r1 = efficient_estiamte_local_energy(qo, s, N)

    ##---given s, estimate H|s>
    # op_mat = torch.stack([PAULI_OPS['I'] for i in range(N)]) ## test
    # tmp = torch.einsum('bnj,nij->bni', s, op_mat) ## test

    # ops, coeff_h = generate_op_list(qo, N)
    # res = torch.einsum("mnij,bnj->bmni", ops, torch.complex(s, torch.zeros_like(s)))
    # c = res[:, :, :, 0]
    # ss = torch.real(res[:, :, :, 1])
    # coeff_p = torch.prod(c, dim=2)
    # c_ = torch.einsum("ij,j->ij", coeff_p, coeff_h)

    ##---given s, estimate <s|H\dag
    op_mat = torch.stack([PAULI_OPS_DAG['I'] for i in range(N)])  ## test
    tmp = torch.einsum('bni,nij->bnj', s, op_mat)  ## test

    ops, coeff_h = generate_op_list(qo, N)
    res = torch.einsum("mnij,bni->bmnj", ops, torch.complex(s, torch.zeros_like(s)))
    c = res[:, :, :, 0]
    ss = torch.real(res[:, :, :, 1])
    coeff_p = torch.prod(c, dim=2)
    c_ = torch.einsum("ij,j->ij", coeff_p, coeff_h)

    qom = get_sparse_operator(qo)
    eigval, eigvec = get_ground_state(qom)

    nz = qom.nonzero()
    for indx, (r, c) in enumerate(zip(nz[0], nz[1])):
        print("{}: ".format(indx), r, c, qom[r, c])

    s = torch.real(res[:, :, :, 1])

    print("Done")


if __name__ == '__main__':
    # A = torch.tensor([[1, 2, 3], [5, 6, 7], [1, 0, 1]])
    # e = torch.einsum("ij,ji", A, A.T)
    # b = torch.tensor([1, 2, 1])
    # c = torch.einsum("ij,j->i", A, b)
    # c_ = torch.matmul(A, b)
    # d = torch.einsum("ii", A)
    # print(torch.allclose(c, c_))
    efficient_estiamte_local_energy_test()
    eloc_test()

    N = 3
    batch = 100
    batch_first = False
    s = generate_random_samples(N, batch, basis=BASIS_VEC, batch_first=batch_first)  # (batch, N, input_dim)
    us = unique_samples(s, batch_first=batch_first)
    sf = generate_full_samples(N, basis=BASIS_VEC, batch_first=False)
    batch_int = vec_to_string(s, N, batch_first=False)

    print(s)
