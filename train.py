import read
from read import MakeLabel

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data as Data
from torch.autograd import Variable
from torch.utils.data import DataLoader 

BATCH_SIZE = 32
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
print(device)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv = nn.Sequential( 
                # 3 * 256 * 256 -> 16 * 128 * 128
                nn.Conv2d(3, 16, 3, padding = 1), 
                nn.Tanh(),
                nn.MaxPool2d(kernel_size = 2),

                nn.Dropout(0.1),

                # 16 * 128 * 128 -> 32 * 64 * 64
                nn.Conv2d(16, 32, 3, padding = 1), 
                nn.Tanh(),
                nn.MaxPool2d(kernel_size = 2),

                nn.Dropout(0.5),

                # 32 * 64 * 64 -> 64 * 32 * 32 
                nn.Conv2d(32, 64, 3, padding = 1), 
                nn.Tanh(),
                nn.MaxPool2d(kernel_size = 2),

                nn.Dropout(0.1),

                # 64 * 32 * 32 -> 128 * 16 * 16 
                nn.Conv2d(64, 128, 3, padding = 1), 
                nn.Tanh(),
                nn.MaxPool2d(kernel_size = 2),

                nn.Dropout(0.1),

                # 128 * 16 * 16 -> 32 * 8 * 8
                nn.Conv2d(128, 32, 3, padding = 1), 
                nn.Tanh(),
                nn.MaxPool2d(kernel_size = 2)
            )

        self.fc = nn.Sequential(
                nn.Linear(32 * 8 * 8, 512),
                nn.Tanh(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.Tanh(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.Tanh(),
                nn.Dropout(0.5),
                nn.Linear(128, 32),
            )

        self.out = nn.Linear(32, 3)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], 4 * 8 * 8, 1, -1).squeeze()
        x = self.fc(x)
        x = self.out(x)
        return x


def MakeLoader(filetype): 
    x, y = MakeLabel(filetype, 256) 
    print('before torch')
    x = torch.tensor(x, dtype = torch.float32)
    y = torch.tensor(y, dtype = torch.long) 
    dataset = Data.TensorDataset(x, y)
    print('after torch')
    dataloader = DataLoader(dataset = dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
    return dataloader

model = CNN().to(device)
print(model)
train_dataloader = MakeLoader('train')
dev_dataloader = MakeLoader('dev')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, betas = (0.9, 0.99))

train_loss, dev_loss = [], []
E_in, E_out = [], []

best_loss = 1e9
print('start training')
for epoch in range(500):
    model.train()
    loss, step = 0, 0
    for (x, y) in train_dataloader:
        step += 1

        x = x.permute(0, 3, 1, 2).to(device)
        y = y.to(device)
        output = model(x)
        tmp_loss = criterion(output, y) 
        loss += tmp_loss.item()

        optimizer.zero_grad()
        tmp_loss.backward()
        optimizer.step()

    loss /= step
    train_loss.append(loss)

    model.eval()
    loss_out, step = 0, 0
    for (x, y) in dev_dataloader:
        step += 1

        x = x.permute(0, 3, 1, 2).to(device)
        y = y.to(device)
        output = model(x)
        tmp_loss = criterion(output, y) 
        loss_out += tmp_loss.item()

    loss_out /= step
    dev_loss.append(loss_out)

    num, acc_in = 0, 0
    for (x, y) in train_dataloader:
        sz = x.size()[0]
        num += sz
        x = x.permute(0, 3, 1, 2).to(device) 
        output = model(x)
        for i in range(sz):
           mx_id = 0 
           for j in range(1, 3):
               if output[i][j] > output[i][mx_id]:
                   mx_id = j
           acc_in += 1 if mx_id == y[i] else 0 

    acc_in /= num 
    acc_in *= 100.0

    num, acc_out = 0, 0
    for (x, y) in dev_dataloader:
        sz = x.size()[0]
        num += sz
        x = x.permute(0, 3, 1, 2).to(device) 
        output = model(x)
        for i in range(sz):
           mx_id = 0 
           for j in range(1, 3):
               if output[i][j] > output[i][mx_id]:
                   mx_id = j
           acc_out += 1 if mx_id == y[i] else 0 

    acc_out /= num 
    acc_out *= 100.0
    if best_loss > loss_out:
        best_loss = loss_out
        torch.save(model.state_dict(), 'model/model{}.pt'.format(epoch + 1))

    print('epoch = {}, train_loss = {:.3f}, train_acc = {:.3f}%, dev_loss = {:.3f}, dev_acc = {:.3f}%'.format(epoch + 1, loss, acc_in, loss_out, acc_out)) 

