import pytest

import numpy as np
import torch
from torch.autograd import Variable

from copy import deepcopy

from test_utils import sample_unit_simplex
from stack import StackRNNModel

def test_policy_network():
    # Inputs
    input_np = np.random.rand(7, 3).astype(np.float32)
    memory_np = np.random.rand(7, 5, 3).astype(np.float32)

    # Pytorch
    input_th = Variable(torch.from_numpy(input_np))
    memory_th = Variable(torch.from_numpy(memory_np))

    model = StackRNNModel(2, 3, stack_depth=5)
    policy_th = model.policy_network(input_th, memory_th)

    assert policy_th.size() == (7, 2 * (5 + 1))

def test_update_stack():
    # Inputs
    memory_np = np.random.rand(7, 5, 3).astype(np.float32)
    p_np = sample_unit_simplex((7, 2 * (5 + 1)), axis=1)
    p_stay_np, p_push_np = p_np[:, :5 + 1], p_np[:, 5 + 1:]
    hidden_np = np.random.rand(7, 3).astype(np.float32)

    # Pytorch
    memory_th = Variable(torch.from_numpy(memory_np))
    p_stay_th = Variable(torch.from_numpy(p_stay_np))
    p_push_th = Variable(torch.from_numpy(p_push_np))
    hidden_th = Variable(torch.from_numpy(hidden_np))

    model = StackRNNModel(2, 3, stack_depth=5)
    new_memory_th = model.update_stack(memory_th, p_stay_th,
                                       p_push_th, hidden_th)

    # Numpy
    new_memory_np = np.zeros((7, 5, 3), dtype=np.float32)
    for b in range(7):
        stack = list(memory_np[b])
        for k in range(5 + 1):
            # Do k pops
            stack_copy = deepcopy(stack[k:])

            # Pack the stack in M_stay
            m_stay = np.asarray(stack_copy)
            # Add to new memory
            if stack_copy:
                new_memory_np[b, :len(stack_copy)] += p_stay_np[b, k] * m_stay

            # Push hidden to the stack
            stack_copy.insert(0, hidden_np[b])
            if len(stack_copy) > 5:
                stack_copy.pop()
            # Pack the stack in M_push
            m_push = np.asarray(stack_copy)
            # Add to new memory
            if stack_copy:
                new_memory_np[b, :len(stack_copy)] += p_push_np[b, k] * m_push

    assert new_memory_th.size() == (7, 5, 3)
    assert np.allclose(new_memory_th.data.numpy(), new_memory_np)
