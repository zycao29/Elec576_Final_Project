import torch
import torch.nn as nn
import torch.nn.functional as F


class model_mlp(nn.Module):
    def __init__(self, input_length, dropout_rate=0.5, is_training=True):
        super(model_mlp,self).__init__()
        self.input_length = input_length;
        self.is_training = is_training;

        self.fc1 = nn.Linear(self.input_length, 500)
        self.bn1 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 5)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = torch.squeeze(x)   
        output =  self.dropout(self.relu(self.bn1(self.fc1(x))))
        output =  self.dropout(self.relu(self.bn2(self.fc2(output))))
        output =  self.fc3(output)
        return output

    
class model_conv(nn.Module):
    def __init__(self, input_length, channel_size=1, dropout_rate=0.5, is_training=True):
        super(model_conv, self).__init__()
        self.input_length = input_length;
        self.channel_size = channel_size;
        self.is_training = is_training;

        kernel_size = 5

        self.conv1 = nn.Conv1d(in_channels=self.channel_size, out_channels=32, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn4 = nn.BatchNorm1d(256)
        #self.avgpool = nn.AvgPool1d(27)
        self.fc = nn.Linear(1536, 5)

        self.pool = nn.MaxPool1d(3)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        batch_size = x.size(0)
        output = self.pool(self.relu( self.bn1(self.conv1(x))));
        output = self.pool(self.relu( self.bn2(self.conv2(output))));
        output = self.pool(self.relu( self.bn3(self.conv3(output))));
        output = self.relu(self.bn4(self.conv4(output)))
        #print(output.shape)
        #output = self.avgpool(output)
        #print(output.shape)
        output = output.view(batch_size, -1)
        #print(output.shape)
        output = self.dropout(output)
        #print(output.shape)
        output = self.fc(output)

        return output
