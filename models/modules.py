import torch
import torch.nn as nn

class TPRencoder_LSTM(nn.Module):
    def __init__(self, encoder_args):
        """
        TPR-LSTM encoder
        """
        for a in encoder_args.keys():
            setattr(self, a, encoder_args[a])

        self.out_dim = self.dSymbols * self.dRoles

        super(TPRencoder_LSTM, self).__init__()

        if self.cell_type == 'LSTM':
            rnn_cls = nn.LSTM
        else:
            rnn_cls = nn.GRU

        self.rnn_aF = rnn_cls(
            self.in_dim, self.out_dim, self.n_layers,
            bidirectional=False,
            dropout=self.dropout,
            batch_first=True)
        self.rnn_aR = rnn_cls(
            self.in_dim, self.out_dim, self.n_layers,
            bidirectional=False,
            dropout=self.dropout,
            batch_first=True)

        self.scale = nn.Parameter(torch.tensor(self.scale_val, dtype=self.get_dtype()), requires_grad=self.train_scale)
        print('self.scale requires grad is: {}'.format(self.scale.requires_grad))

        self.ndirections = 1 + int(self.bidirectional)
        self.F = nn.Linear(self.nSymbols, self.dSymbols)
        self.R = nn.Linear(self.nRoles, self.dRoles)
        self.WaF = nn.Linear(self.out_dim, self.nSymbols)
        self.WaR = nn.Linear(self.out_dim, self.nRoles)
        self.softmax = nn.Softmax(dim=2)

    def get_dtype(self):
        return next(self.parameters()).dtype

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.n_layers, batch, self.out_dim)
        if self.cell_type == 'LSTM':
            return (weight.new(*hid_shape).zero_(), weight.new(*hid_shape).zero_())
        else:
            return weight.new(*hid_shape).zero_()

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        seq = x.size(1)
        hidden_aF = self.init_hidden(batch)  # includes both hidden state and cell state (h_n,c_n).
        hidden_aR = self.init_hidden(batch)
        if self.training:
            self.rnn_aF.flatten_parameters()
            self.rnn_aR.flatten_parameters()
        for i in range(seq):
            aF, hidden_aF = self.rnn_aF(x[:, [i], :], hidden_aF)
            aR, hidden_aR = self.rnn_aR(x[:, [i], :], hidden_aR)
            aF = self.WaF(aF)
            aR = self.WaR(aR)
            aF = self.softmax(aF / self.temperature)
            aR = self.softmax(aR / self.temperature)
            itemF = self.F(aF)
            itemR = self.R(aR)
            T = (torch.bmm(torch.transpose(itemF, 1, 2), itemR)).view(batch, -1)

            if self.cell_type == 'LSTM':
                hidden_aF = (T.unsqueeze(0), hidden_aF[1])
                hidden_aR = (T.unsqueeze(0), hidden_aR[1])
            else:
                hidden_aF = T.unsqueeze(0)
                hidden_aR = T.unsqueeze(0)
            if i == 0:
                out = T.unsqueeze(1)
                aFs = aF
                aRs = aR
            else:
                out = torch.cat([out, T.unsqueeze(1)], 1)
                aFs = torch.cat([aFs, aF], 1)
                aRs = torch.cat([aRs, aR], 1)

        return out, aFs, aRs

    def backward(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        seq = x.size(1)
        hidden_aF = self.init_hidden(batch)  # includes both hidden state and cell state (h_n,c_n).
        hidden_aR = self.init_hidden(batch)
        if self.training:
            self.rnn_aF.flatten_parameters()
            self.rnn_aR.flatten_parameters()
        for i in range(seq - 1, -1, -1):
            aF, hidden_aF = self.rnn_aF(x[:, [i], :], hidden_aF)
            aR, hidden_aR = self.rnn_aR(x[:, [i], :], hidden_aR)
            aF = self.WaF(aF)
            aR = self.WaR(aR)
            aF = self.softmax(aF / self.temperature)
            aR = self.softmax(aR / self.temperature)
            itemF = self.F(aF)
            itemR = self.R(aR)

            T = (torch.bmm(torch.transpose(itemF, 1, 2), itemR)).view(batch, -1)

            if self.cell_type == 'LSTM':
                hidden_aF = (T.unsqueeze(0), hidden_aF[1])
                hidden_aR = (T.unsqueeze(0), hidden_aR[1])
            else:
                hidden_aF = T.unsqueeze(0)
                hidden_aR = T.unsqueeze(0)
            if i == seq - 1:
                out = T.unsqueeze(1)
                aFs = aF
                aRs = aR
            else:
                out = torch.cat([T.unsqueeze(1), out], 1)
                aFs = torch.cat([aF, aFs], 1)
                aRs = torch.cat([aR, aRs], 1)

        return out, aFs, aRs

    def call(self, x):

        R_flat = self.R.weight.view(self.R.weight.shape[0], -1)
        R_loss_mat = torch.norm(torch.mm(R_flat, R_flat.t()) - torch.eye(R_flat.shape[0], dtype=R_flat.dtype, device=R_flat.device)).pow(2) +\
                     torch.norm(torch.mm(R_flat.t(), R_flat) - torch.eye(R_flat.shape[1], dtype=R_flat.dtype, device=R_flat.device)).pow(2)
        R_loss = torch.norm(R_loss_mat)

        if self.ndirections == 1:
            out, aFs, aRs = self.backward(x)
            return out, (out[:, 0, :], aFs, aRs), R_loss

        else:
            out_f, aFs_f, aRs_f = self.forward(x)
            out_b, aFs_b, aRs_b = self.backward(x)

            out = torch.cat((out_f[:, :, :], out_b[:, :, :]), dim=-1)
            last_out = torch.cat((out_f[:, -1, :], out_b[:, 0, :]), dim=-1)
            aFs = torch.stack((aFs_f, aFs_b), dim=-1)
            aRs = torch.stack((aRs_f, aRs_b), dim=-1)
            return out, (last_out, aFs, aRs), R_loss
