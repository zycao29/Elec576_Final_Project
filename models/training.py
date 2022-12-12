import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import copy
import random
from collections import OrderedDict
from models.model import *
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix



def train_model(model, dataloaders, criterion, optimizer, lr_scheduler, device, args):
    
    for epoch in range(args.num_epoches):
        print("Epoch {}/{}".format(epoch, args.num_epoches-1));
        print('+' * 80)

        train_losses = []
        train_true_labels = []
        train_pred_labels = []

        model.train()
        for x, labels in tqdm(dataloaders['train']):

            x = x.to(device, dtype=torch.float32)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_pred_labels.append(logits.detach().cpu())
            train_true_labels.append(labels.detach().cpu())

        lr_scheduler.step()
        all_pred = np.vstack(train_pred_labels)
        all_true = np.vstack(train_true_labels)
        # convert from one-hot coding to binary label.
        all_pred_binary = np.argmax(all_pred, axis=1)
        all_true_binary = np.argmax(all_true, axis=1)
        print("                         Training:")
        print("Loss: %.4f" %(np.mean(np.array(train_losses))))
        ACC = accuracy_score(all_true_binary, all_pred_binary)
        print("Accuracy: %.4f " %(ACC))
        print(confusion_matrix(all_true_binary, all_pred_binary))
    torch.save(model.state_dict(), os.path.join(args.model_para, 'model.std'))

    print("#"*20)
    # Testing
    model.load_state_dict(torch.load(os.path.join(args.model_para, 'model.std')))

    test_losses = []
    test_pred_labels = []
    test_true_labels = []
    model.eval()
    for x, labels in tqdm(dataloaders['test']):
        x = x.to(device, dtype=torch.float32)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits,labels)
        test_losses.append(loss.item())
        test_true_labels.append(labels.detach().cpu())
        test_pred_labels.append(logits.detach().cpu())

    all_pred = np.vstack(test_pred_labels)
    all_true = np.vstack(test_true_labels)
    all_pred_binary = np.argmax(all_pred, axis=1)
    all_true_binary = np.argmax(all_true, axis=1)
    print("                         Testing:")
    print("Loss: %.4f" %(np.mean(np.array(test_losses))))
    print("Accuracy: %.4f " %(accuracy_score(all_true_binary, all_pred_binary)))
    print(confusion_matrix(all_true_binary, all_pred_binary))



def train_model_fine_tune(model, dataloaders, criterion, optimizer, lr_scheduler, device, args):
    
    model_cnnf = model_conv(187)
    model_cnnf.load_state_dict(torch.load(os.path.join(args.model_para, 'model.std')))
    model_state_dict = model.state_dict()
    temp = OrderedDict()
    model_state_dict = model.state_dict(destination=None)
    for name, param in model_cnnf.named_parameters():
        if name in model_state_dict:
            temp[name] = param

    model_state_dict.update(temp)
    model.load_state_dict(model_state_dict)

    for epoch in range(args.num_epoches):
        print("Epoch {}/{}".format(epoch, args.num_epoches-1));
        print('+' * 80)

        train_losses = []
        train_true_labels = []
        train_pred_labels = []

        model.train()
        for x, labels in tqdm(dataloaders['train']):

            x = x.to(device, dtype=torch.float32)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_pred_labels.append(logits.detach().cpu())
            train_true_labels.append(labels.detach().cpu())

        lr_scheduler.step()
        all_pred = np.vstack(train_pred_labels)
        all_true = np.vstack(train_true_labels)
        # convert from one-hot coding to binary label.
        all_pred_binary = np.argmax(all_pred, axis=1)
        all_true_binary = np.argmax(all_true, axis=1)
        print("                         Training:")
        print("Loss: %.4f" %(np.mean(np.array(train_losses))))
        ACC = accuracy_score(all_true_binary, all_pred_binary)
        print("Accuracy: %.4f " %(ACC))
        print(confusion_matrix(all_true_binary, all_pred_binary))
    torch.save(model.state_dict(), os.path.join(args.model_para, 'model.std'))

    print("#"*20)
    # Testing
    model.load_state_dict(torch.load(os.path.join(args.model_para, 'model.std')))

    test_losses = []
    test_pred_labels = []
    test_true_labels = []
    model.eval()
    for x, labels in tqdm(dataloaders['test']):
        x = x.to(device, dtype=torch.float32)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits,labels)
        test_losses.append(loss.item())
        test_true_labels.append(labels.detach().cpu())
        test_pred_labels.append(logits.detach().cpu())

    all_pred = np.vstack(test_pred_labels)
    all_true = np.vstack(test_true_labels)
    all_pred_binary = np.argmax(all_pred, axis=1)
    all_true_binary = np.argmax(all_true, axis=1)
    print("                         Testing:")
    print("Loss: %.4f" %(np.mean(np.array(test_losses))))
    print("Accuracy: %.4f " %(accuracy_score(all_true_binary, all_pred_binary)))
    print(confusion_matrix(all_true_binary, all_pred_binary))