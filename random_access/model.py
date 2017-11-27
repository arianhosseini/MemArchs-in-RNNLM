import torch
import torch.nn as nn
from torch.autograd import Variable

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, k=3):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTMCell(ninp, nhid)
        self.decoder = nn.Linear(nhid, ntoken)

        self.w_mi = nn.Linear(nhid, 1)
        self.w_mh = nn.Linear(nhid, 1)
        self.w_hh = nn.Linear(nhid, nhid)
        self.w_hm = nn.Linear(nhid, nhid)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.k = k

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

        self.w_mh.weight.data.uniform_(-initrange, initrange)
        self.w_mi.weight.data.uniform_(-initrange, initrange)
        self.w_hh.weight.data.uniform_(-initrange, initrange)
        self.w_hm.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        outputs = []
        t = 0
        h_t, c_t = hidden
        hiddens = [h_t]
        softmax = nn.Softmax()
        # print h_t.size()

        for step in range(emb.size()[0]):
            h_t, c_t = self.rnn(emb[step], (h_t, c_t))
            if t >= self.k:

                e_is = []
                e_t = self.w_mh(h_t) #bs x 1
                for h_i in hiddens[-self.k:]:
                    e_is.append(self.w_mi(h_i) + e_t) #[bs x 1]
                att_pre_weights = torch.cat(e_is, dim=1) # bs x k
                att_weights = softmax(att_pre_weights)
                hts = torch.stack(hiddens[-self.k:]).permute(1,0,2) #bs x k x nhid
                m_t = torch.mul(hts,att_weights.unsqueeze(2)).sum(1) #bs x nhid
                new_h_t = self.w_hh(h_t) + self.w_hm(m_t)
                outputs.append(new_h_t)
                # print "newh shape: ", new_h_t.size()
            else:
                outputs.append(h_t)

            t += 1
        output = torch.stack(outputs, 0)

        # output, hidden = self.rnn(emb, hidden)

        # print output.size()
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(bsz, self.nhid).zero_()),
                    Variable(weight.new(bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
