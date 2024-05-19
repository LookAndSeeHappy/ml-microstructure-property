from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import copy
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
plt.ion()

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(340),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize([340, 340]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# the path of window images
data_dir = 'YOUR_PATH'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, scheduler, num_epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            y_labels = []
            y_pred = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                for i in labels.data.cpu().numpy():
                    y_labels.append(i)
                for j in preds.cpu().numpy():
                    y_pred.append(j)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            a = precision_score(y_labels, y_pred, average=None)
            b = recall_score(y_labels, y_pred, average=None)
            if phase == 'train':
                writer_train.add_scalar("loss", epoch_loss, epoch)
                writer_train.add_scalar("acc", epoch_acc, epoch)
                writer_train.add_scalar("precision_score_low",a[0], epoch)
                writer_train.add_scalar("recall_score_low",b[0], epoch)
                writer_train.add_scalar("precision_score_median", a[1], epoch)
                writer_train.add_scalar("recall_score_median", b[1], epoch)
                writer_train.add_scalar("precision_score_high", a[2], epoch)
                writer_train.add_scalar("recall_score_high", b[2], epoch)
            if phase == 'val':
                writer_val.add_scalar("loss", epoch_loss, epoch)
                writer_val.add_scalar("acc", epoch_acc, epoch)
                writer_val.add_scalar("precision_score_low", a[0], epoch)
                writer_val.add_scalar("recall_score_low", b[0], epoch)
                writer_val.add_scalar("precision_score_median", a[1], epoch)
                writer_val.add_scalar("recall_score_median", b[1], epoch)
                writer_val.add_scalar("precision_score_high", a[2], epoch)
                writer_val.add_scalar("recall_score_high", b[2], epoch)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), "best{}.pth".format(epoch))
        print()
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, "best.pth")
    return model


writer_train = SummaryWriter("./logs/train")
writer_val = SummaryWriter("./logs/val")
model_conv = torchvision.models.resnet18(pretrained=True)

num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 3)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()
lr=0.0001

fc_params = list(map(id, model_conv.fc.parameters()))
base_params = filter(lambda p: id(p) not in fc_params, model_conv.parameters())
params = [{'params': base_params},
          {'params': model_conv.fc.parameters(), 'lr': lr * 10}]
optimizer_conv = optim.SGD(params, lr=lr)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=1200, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=1600)
writer_train.close()
writer_val.close()
