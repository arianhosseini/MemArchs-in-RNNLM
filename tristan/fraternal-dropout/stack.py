import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from embed_regularize import fixMaskEmbeddedDropout
from weight_drop import WeightDrop
from model import fixMaskDropout

class StackRNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, dropout=0.5, dropouti=0.5,
                 dropoute=0.1, wdrop=0, tie_weights=False, stack_depth=10):
        super(StackRNNModel, self).__init__()
        self.idrop = fixMaskDropout(dropouti)
        self.drop = fixMaskDropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp, padding_idx=0)
        self.embedded_dropout = fixMaskEmbeddedDropout(self.encoder, dropoute)

        # self.lstm = WeightDrop(torch.nn.LSTM(ninp, ninp), ['weight_hh_l0'], dropout=wdrop)
        self.lstm_cell = nn.LSTMCell(ninp, ninp)

        self.decoder = nn.Linear(ninp, ntoken)
        self.decoder.weight = self.encoder.weight_raw

        self.policy_network_stack = nn.Conv1d(ninp, 2, kernel_size=2)
        self.policy_network_input = nn.Linear(ninp, 2 * (stack_depth + 1))

        self.init_weights()

        self.ninp = ninp
        self.dropoute = dropoute
        self.stack_depth = stack_depth

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight_raw.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def policy_network(self, input, memory):
        memory_padded = F.pad(memory.transpose(1, 2), (0, 2), 'constant', 0)
        policy_stack = self.policy_network_stack(memory_padded)
        policy_stack = policy_stack.view(input.size(0), -1)
        policy_input = self.policy_network_input(input)

        return F.softmax(policy_stack + policy_input, dim=1)

    def update_stack(self, memory, p_stay, p_push, hidden):
        p_stay = p_stay.unsqueeze(2).unsqueeze(3)
        p_push = p_push.unsqueeze(2).unsqueeze(3)

        memory_padded = F.pad(memory.unsqueeze(1),
            (0, 0, 0, self.stack_depth), 'constant', 0)
        kernel = Variable(torch.eye(self.stack_depth + 1).view(
            self.stack_depth + 1, 1, self.stack_depth + 1, 1),
            requires_grad=False)
        m_stay = F.conv2d(memory_padded, kernel)

        pushed = (hidden.unsqueeze(1).clone()
                        .repeat(1, self.stack_depth + 1, 1)
                        .unsqueeze(2))
        m_push = torch.cat([pushed, m_stay[:, :, :-1]], dim=2)

        return (torch.sum(m_stay * p_stay, dim=1)
                + torch.sum(m_push * p_push, dim=1))

    def forward(self, inputs, hidden, memory, return_h=False, draw_mask_e=True, draw_mask_i=True, draw_mask_w=True, draw_mask_o=True):
        emb = self.embedded_dropout(draw_mask_e, inputs)
        
        emb_i = self.idrop(draw_mask_i, emb)

        raw_output_list, output_list = list(), list()
        for input in emb_i:
            hidden = self.lstm_cell(input, hidden)
            output = self.drop(draw_mask_o, hidden[0])

            raw_output_list.append(hidden[0])
            output_list.append(output)

            policy = self.policy_network(input, memory)
            p_stay, p_push = torch.chunk(policy, 2, dim=1)
            memory = self.update_stack(memory, p_stay, p_push, hidden[0])

        raw_output = torch.stack(raw_output_list, dim=0)
        output = torch.stack(output_list, dim=0)

        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        result = decoded.view(output.size(0), output.size(1), decoded.size(1))

        if return_h:
            return result, hidden, memory, raw_output, output

        return result, hidden, memory

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(bsz, self.ninp).zero_()),
                Variable(weight.new(bsz, self.ninp).zero_()))

    def init_memory(self, bsz):
        weight = next(self.parameters()).data
        return Variable(weight.new(bsz, self.stack_depth, self.ninp).zero_())
