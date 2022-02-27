import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from transformer import VisionTransformer
from MLP_mixer import Mixer
'''Data Loading'''
dataset_name = 'OPPORTUNITY'
X_train = torch.from_numpy(np.load('/mnt/experiment/xushige/public_dataset/%s/x_train.npy'%(dataset_name))).float()
X_test = torch.from_numpy(np.load('/mnt/experiment/xushige/public_dataset/%s/x_test.npy'%(dataset_name))).float()
Y_train = torch.from_numpy(np.load('/mnt/experiment/xushige/public_dataset/%s/y_train.npy'%(dataset_name))).long()
Y_test = torch.from_numpy(np.load('/mnt/experiment/xushige/public_dataset/%s/y_test.npy'%(dataset_name))).long()

'''Data Ascending Dimension'''
X_train = X_train.unsqueeze(dim=1)
X_test = X_test.unsqueeze(dim=1)
print('X_train shape: %s\nX_test shape: %s\nY_train shape: %s\nY_test shape: %s' % (X_train.size(), X_test.size(), Y_train.size(), Y_test.size()))
n_classes = len(set(Y_test))
print('n_classes: %d\n' % n_classes)

'''Dataset Build'''
train_data = TensorDataset(X_train, Y_train)
test_data = TensorDataset(X_test, Y_test)

'''DataLoader Build'''
EP = 100
B_S = 128
train_loader = DataLoader(train_data, batch_size=B_S, shuffle=True)
test_loader = DataLoader(test_data, batch_size=B_S, shuffle=True)

'''Model'''
## MLP_Mixer model
model = Mixer(
    num_classes=n_classes, 
    image_size=X_train.shape[-2:], 
    patch_size=(1, 9), 
    num_layers=8,
    embed_dim=512,
    ds=256,
    dc=2048,
    drop_rate=0).cuda()

## VisionTransformer model
model = VisionTransformer(
        patch_size=[16, 25], 
        img_size=X_train.shape[-2:],
        in_channels=1,
        hidden_size=512,
        layer_num=16,
        head_num=16,
        mlp_dim=1024,
        num_classes=n_classes,
        drop_rate=0.1,
        zero_head=False,
        vis=True
        ).cuda()
LR = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.001)
lr_schedual = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.5)
loss_fn = nn.CrossEntropyLoss()

'''Train Process'''
for i in range(EP):
    for data, label in train_loader:
        data, label = data.cuda(), label.cuda()
        model.train()
        out = model(data)
        loss = loss_fn(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    lr_schedual.step()
    cor_num = 0
    for data, label in test_loader:
        data, label = data.cuda(), label.cuda()
        model.eval()
        out = model(data)
        _, pre = torch.max(out, dim=1)
        cor_num += (pre == label).sum()
    acc = cor_num.item() / len(test_data)
    print('====================================================\nEpoch: %d   Loss: %f   Acc: %f'%(i, loss, acc))
