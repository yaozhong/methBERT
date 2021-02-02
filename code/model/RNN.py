import torch
import torch.nn as nn
import torch.nn.functional as F

# sequence classification model
# fixed baseline model as in the deepMOD
class biRNN_basic(nn.Module):
    def __init__(self, device, input_size=7, hidden_size=100, num_layers=3, num_classes=2):
        super(biRNN_basic, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # bi-directional LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        # self.sigmoid = nn.Sigmoid()
        # self.tanh = nn.Tanh()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)

        #out, _ = self.lstm(self.embed(x), (h0, c0))
        rnn_out, _ = self.lstm(x, (h0, c0))
        
        # linear-1x
        out = self.fc(rnn_out[:, int(x.size(1)/2) ,:])

        # linear-7x
        """
        mid_idx = int(x.size(1)/2)
        out = self.fc(torch.cat((rnn_out[:, mid_idx-3, :], rnn_out[:, mid_idx-2, :], rnn_out[:, mid_idx-1, :], rnn_out[:, mid_idx, :], 
            rnn_out[:, mid_idx+1, :], rnn_out[:, mid_idx+2, :], rnn_out[:, mid_idx+3, :]), -1))
        out = self.tanh(out)
        """

        return out

# current one test the positional shift
class biRNN_test(nn.Module):
    def __init__(self, device, input_size=7, hidden_size=100, num_layers=3, num_classes=2):
        super(biRNN_test, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # bi-directional LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)

        #out, _ = self.lstm(self.embed(x), (h0, c0))
        rnn_out, _ = self.lstm(x, (h0, c0))
        out = self.fc(rnn_out[:, int(x.size(1)/2)+1 ,:])

        return out


class biRNN(nn.Module):
    def __init__(self, device, input_size=7, hidden_size=100, num_layers=3, num_classes=2):
        super(biRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # bi-directional LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        self.fc0 = nn.Linear(hidden_size*2, 32)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)

        rnn_out, _ = self.lstm(x, (h0, c0))
        
        out = F.relu(self.fc0(rnn_out[:, int(x.size(1)/2) ,:]))
        out = self.fc(out)

        return out

# 2020/08/31
class biRNN_test_embed(nn.Module):
    def __init__(self, device, input_size=7, hidden_size=100, num_layers=3, num_classes=2):
        super(biRNN_test_embed, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # bi-directional LSTM
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.featEmbed = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)

        #out, _ = self.lstm(self.embed(x), (h0, c0))
        rnn_out, _ = self.lstm(self.featEmbed(x), (h0, c0))
        out = self.fc(rnn_out[:, int(x.size(1)/2) ,:])

        return out

# add residual 
# residual augmented implementation of ElMo
class biRNN_residual(nn.Module):
    def __init__(self, device, input_size=7, hidden_size=100, num_layers=3, num_classes=2):
        super(biRNN_residual, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # bi-directional LSTM
        self.lstm1 = nn.LSTM(input_size, hidden_size, 1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size*2 + input_size, hidden_size, 1, batch_first=True, bidirectional=True)
        
        self.fc = nn.Linear(607, 128)
        self.fc2 =nn.Linear(128, num_classes)

    def forward(self, x):

        # different layer implementation
        rep = [x[:, int(x.size(1)/2) ,:]]
        #h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)
        #c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)

        # initial layer processing
        h, _ = self.lstm1(x)
        rep.append(h[:, int(x.size(1)/2) ,:])

        for i in range(1, self.num_layers):

            ch =  torch.cat([h, x], -1)
            h, _ = self.lstm2(ch)

            rep.append(h[:, int(x.size(1)/2) ,:])

        # first dimention is the sample
        rep = torch.cat(rep, dim=-1)

        out = self.fc(rep)
        out = self.fc2(out)

        return out    
