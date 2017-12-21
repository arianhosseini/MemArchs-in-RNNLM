import torch
import torch.nn as nn
from torch.autograd import Variable

from embed_regularize import fixMaskEmbeddedDropout
from weight_drop import WeightDrop

class fixMaskDropout(nn.Module):
    def __init__(self, dropout=0.5):
        super(fixMaskDropout, self).__init__()
        self.dropout = dropout
        self.mask = None

    def forward(self, draw_mask, input):
        if self.training == False:
            return input
        if self.mask is None or draw_mask==True:
            self.mask =  input.data.new().resize_(input.size()).bernoulli_(1 - self.dropout) / (1 - self.dropout)
        mask = Variable(self.mask)
        masked_input = mask*input
        return masked_input

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, dropout=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, k=0):
        super(RNNModel, self).__init__()
        self.idrop = fixMaskDropout(dropouti)
        self.drop = fixMaskDropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp, padding_idx=0)
        self.embedded_dropout = fixMaskEmbeddedDropout(self.encoder, dropoute)
        self.lstm_cell = WeightDrop(torch.nn.LSTMCell(ninp, ninp), ['weight_hh'], dropout=wdrop)
        # self.lstm = WeightDrop(torch.nn.LSTM(ninp, ninp), ['weight_hh_l0'], dropout=wdrop)
        self.decoder = nn.Linear(ninp, ntoken)
        self.decoder.weight = self.encoder.weight_raw
        self.ninp = ninp
        self.dropoute = dropoute
        self.k = k
        if k > 0:
            self.w_mi = nn.Linear(ninp, 1)
            self.w_mh = nn.Linear(ninp, 1)
            self.w_hh = nn.Linear(ninp, ninp)
            self.w_hm = nn.Linear(ninp, ninp)


        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight_raw.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

        if self.k > 0:
            self.w_mh.weight.data.uniform_(-initrange, initrange)
            self.w_mi.weight.data.uniform_(-initrange, initrange)
            self.w_hh.weight.data.uniform_(-initrange, initrange)
            self.w_hm.weight.data.uniform_(-initrange, initrange)



    def forward(self, input, hidden, return_h=False, draw_mask_e=True, draw_mask_i=True, draw_mask_w=True, draw_mask_o=True):
        emb = self.embedded_dropout(draw_mask_e, input)
        t = 0
        emb_i = self.idrop(draw_mask_i, emb)
        softmax = nn.Softmax()
        h_t, c_t = hidden
        raw_output = []
        hiddens = [h_t]

        for t in range(emb.size(0)):
            h_t, c_t = self.lstm_cell(draw_mask_w, emb_i[t], (h_t,c_t))
            if self.k > 0:
                e_is = []
                e_t = self.w_mh(h_t) #bs x 1
                for h_i in hiddens[-self.k:]:
                    e_is.append(self.w_mi(h_i) + e_t) #[bs x 1]
                att_pre_weights = torch.cat(e_is, dim=1) # bs x k
                att_weights = softmax(att_pre_weights)
                hts = torch.stack(hiddens[-self.k:]).permute(1,0,2) #bs x k x nhid
                m_t = torch.mul(hts,att_weights.unsqueeze(2)).sum(1) #bs x nhid
                new_h_t = self.w_hh(h_t) + self.w_hm(m_t)
                raw_output.append(new_h_t)
                hiddens.append(h_t)
                # print "newh shape: ", new_h_t.size()
            else:
                new_h_t = h_t
                raw_output.append(h_t)


        raw_output = torch.stack(raw_output).squeeze()
        hidden = (new_h_t, c_t)
        # print "raw output size: ", raw_output.size()
        # raw_output, hidden = self.lstm(draw_mask_w, emb_i, hidden)
        output = self.drop(draw_mask_o, raw_output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        result = decoded.view(output.size(0), output.size(1), decoded.size(1))

        if return_h:
            return result, hidden, raw_output, output
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(bsz, self.ninp).zero_()),
                Variable(weight.new(bsz, self.ninp).zero_()))
