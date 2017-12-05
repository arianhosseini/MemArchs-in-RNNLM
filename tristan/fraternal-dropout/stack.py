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
        self.lstm = WeightDrop(torch.nn.LSTM(ninp, ninp), ['weight_hh_l0'], dropout=wdrop)
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

        return policy_stack + policy_input

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

    def forward(self, input, hidden, return_h=False, draw_mask_e=True, draw_mask_i=True, draw_mask_w=True, draw_mask_o=True):
        emb = self.embedded_dropout(draw_mask_e, input)
        
        emb_i = self.idrop(draw_mask_i, emb)

        raw_output, hidden = self.lstm(draw_mask_w, emb_i, hidden)
        output = self.drop(draw_mask_o, raw_output)

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        result = decoded.view(output.size(0), output.size(1), decoded.size(1))

        if return_h:
            return result, hidden, raw_output, output
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(1, bsz, self.ninp).zero_()),
                Variable(weight.new(1, bsz, self.ninp).zero_()))
