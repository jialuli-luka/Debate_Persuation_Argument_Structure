import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

torch.set_default_tensor_type('torch.FloatTensor')


class model(nn.Module):

    def __init__(self,embedding_dim = 768, hidden_dim = 32):
        super(model, self).__init__()

        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.fully_connected = nn.Linear(hidden_dim*2, 2)
        self.dropout = nn.Dropout(p=0.5)
        self.loss = nn.CrossEntropyLoss()

    def compute_Loss(self, pred_vec, gold_seq):
        return self.loss(pred_vec, gold_seq)

    def saveModels(self):
        torch.save(self.state_dict(), 'my_trained_model_weights')

    def forward(self, input):
        input_vector = torch.tensor(input,dtype=torch.float64).unsqueeze(1)
        rnn_output,_ = self.rnn(input_vector)
        rnn_output = rnn_output.view(rnn_output.shape[0], rnn_output.shape[1], 2, self.hidden_dim)
        fc_input = torch.cat([rnn_output[-1,:,0,:].unsqueeze(0), rnn_output[0,:,1,:].unsqueeze(0)], 2)
        output = F.softmax(self.fully_connected(self.dropout(fc_input)),dim = 2)

        return output


class combine_model(nn.Module):

    def __init__(self,embedding_dim = 768, hidden_dim = 32, arg_hidden_dim = 4):
        super(combine_model, self).__init__()

        self.hidden_dim = hidden_dim
        self.arg_hidden_dim = arg_hidden_dim
        self.rnn_bert = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.rnn_arg = nn.LSTM(32, arg_hidden_dim, bidirectional=True)
        self.fully_connected_bert = nn.Linear(hidden_dim*2, 2)
        self.fully_connected_arg = nn.Linear(arg_hidden_dim*2, 2)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.loss = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.tensor([0.5]),requires_grad=True)

    def compute_Loss(self, pred_vec, gold_seq):
        return self.loss(pred_vec, gold_seq)

    def saveModels(self):
        torch.save(self.state_dict(), 'my_trained_model_weights')

    def forward(self, input_bert, input_arg):
        input_vector_bert = torch.tensor(input_bert,dtype=torch.float64).unsqueeze(1)
        input_vector_arg = torch.tensor(input_arg,dtype=torch.float64).unsqueeze(1)
        rnn_output_bert,_ = self.rnn_bert(input_vector_bert)
        rnn_output_arg,_ = self.rnn_arg(input_vector_arg)
        rnn_output_bert = rnn_output_bert.view(rnn_output_bert.shape[0], rnn_output_bert.shape[1], 2, self.hidden_dim)
        rnn_output_arg = rnn_output_arg.view(rnn_output_arg.shape[0], rnn_output_arg.shape[1], 2, self.arg_hidden_dim)
        fc_input_bert = torch.cat([rnn_output_bert[-1,:,0,:].unsqueeze(0), rnn_output_bert[0,:,1,:].unsqueeze(0)], 2)
        fc_input_arg = torch.cat([rnn_output_arg[-1,:,0,:].unsqueeze(0), rnn_output_arg[0,:,1,:].unsqueeze(0)], 2)
        output_bert = F.softmax(self.fully_connected_bert(self.dropout1(fc_input_bert)),dim = 2)
        output_arg = F.softmax((self.fully_connected_arg(self.dropout2(fc_input_arg))),dim = 2)
        output0 = torch.add(torch.mul(output_bert[0,0,0],self.weight),torch.mul(output_arg[0,0,0],torch.add(1,-self.weight)))
        output1 = torch.add(torch.mul(output_bert[0,0,1],self.weight),torch.mul(output_arg[0,0,1],torch.add(1,-self.weight)))
        output = torch.cat([output0.unsqueeze(0).unsqueeze(0),output1.unsqueeze(0).unsqueeze(0)],2)
        return output
