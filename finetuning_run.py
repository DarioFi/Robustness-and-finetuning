#!/usr/bin/env python
# coding: utf-8

# # DISTANCE CODE


import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision

import numpy as np
import matplotlib.pyplot as plt

import wandb
import json

from utils import *


def train(dataloader, model, loss_fn, optimizer, initial_parameters, Correlations, epoch, ALL_par):
    size = len(dataloader.dataset)  # the .dataset attribute gives us the original data, so this is 60000
    num_batches = len(
        dataloader)  # the dataloader is an iterable that produces batchs, so its length is the num. of batches
    model.train()
    mean_loss = 0.0
    correct = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if (batch + 1) % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Step [{batch + 1}/{num_batches}], Loss {loss:.4f}")

        if (batch + 1) % 100 == 0:
            step = (num_batches * epoch) + (batch + 1)
            current_parameters = []

            with torch.no_grad():

                for name, param in model.named_parameters():
                    if ALL_par:
                        if ('conv' in name) or ('fc' in name):
                            current_parameters.append(param.clone())
                    else:
                        current_parameters.append(param.clone())

                generate_corr(current_parameters, initial_parameters, step, Correlations)
                print(f"Decorrelation from initial weights : {Correlations[-1]}")

    ## Compute and print some data to monitor the training
    mean_loss /= num_batches
    correct /= size
    print(f" \nTRAINING - Accuracy: {(100 * correct):>5.1f}%, Avg loss: {mean_loss:>7f}")
    return mean_loss, correct


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0.0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"TESTING  - Accuracy: {(100 * correct):>5.1f}%, Avg loss: {test_loss:>8f}\n")
    return test_loss, correct


def corr_fun(current_parameters, initial_parameters, t):
    par_0 = initial_parameters
    par_t = current_parameters
    M = sum([torch.numel(par) for par in par_0])

    with torch.no_grad():
        corr = 0
        for i in range(len(par_0)):
            w_0 = torch.Tensor(par_0[i])
            w_t = torch.Tensor(par_t[i])

            square = (w_0 - w_t) ** 2
            corr += torch.sum(square)

        result = corr / M
        return result


def generate_corr(current_parameters, initial_parameters, t, Correlations):
    with torch.no_grad():
        if t >= 0:

            result = corr_fun(current_parameters, initial_parameters, t)
            Correlations.append(float(result))
            wandb.log({'Correlation Function wrt initial parameters': result}, step=t)

        else:
            Correlations.append('\\')


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# login to wandb
wandb.login(key="aaaa6a0838d0e4f2d3f68ffbc76dcb40602660cf", relogin=True, force=False)  # mattia

iters = 2

for lr in [0.001, 0.00005]:

    for index in range(0, iters):
        CLASSES = 10

        train_set, test_set = load_cifar(CLASSES, size=(112, 112))

        batch_size = 128
        train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_dl = DataLoader(test_set, batch_size=batch_size)

        ##Build the 3 models

        # Vanilla

        model_vanilla = torchvision.models.resnet18(num_classes=50)
        model_vanilla.load_state_dict(torch.load("good_pretrained_models/resnet_vanilla"))
        #
        num_classes = CLASSES
        model_vanilla.fc = nn.Linear(512, num_classes, bias=True, device=device)
        model_vanilla.to(device)
        #
        # Noise

        model_noise = torchvision.models.resnet18(num_classes=50)
        model_noise.load_state_dict(torch.load("good_pretrained_models/resnet_noise"))

        num_classes = CLASSES
        model_noise.fc = nn.Linear(512, num_classes, bias=True, device=device)
        model_noise.to(device)

        # SWA

        model_swa = torchvision.models.resnet18(num_classes=50)
        model_name = "good_pretrained_models/resnet_swa"

        sd = torch.load(model_name)
        if "swa" in model_name:
            ks = list(sd.keys())
            for key in ks:
                if "module." not in key:
                    continue

                nk = key.replace("module.", "")
                sd[nk] = sd[key]
                del sd[key]
            del sd["n_averaged"]

        model_swa.load_state_dict(sd)

        num_classes = CLASSES
        model_swa.fc = nn.Linear(512, num_classes, bias=True, device=device)
        model_swa.to(device)

        # In[14]:

        for model in [model_vanilla, model_noise, model_swa]:
            for name, param in model.named_parameters():
                # resnet
                param.requires_grad = True

        # In[15]:

        optimizer_vanilla = torch.optim.AdamW(model_vanilla.parameters(), lr=lr)
        optimizer_noise = torch.optim.AdamW(model_noise.parameters(), lr=lr)
        optimizer_swa = torch.optim.AdamW(model_swa.parameters(), lr=lr)

        loss_fn = torch.nn.CrossEntropyLoss()

        # ## TEST VANILLA

        # In[16]:

        epochs = 15

        config = {
            "learning_rate": 0.0001,
            "architecture": "resnet18_vanilla",
            "dataset": f"cifar{CLASSES}",
            "epochs": epochs,
            "lr": lr
        }

        # initiate run
        run = wandb.init(project="ML-project-Correlations",
                         config=config, reinit=True, save_code=True)

        Correlations_vanilla = []
        ALL_par = True
        current_parameters = []
        for name, param in model_vanilla.named_parameters():
            if ALL_par:
                if ('conv' in name) or ('fc' in name):
                    current_parameters.append(param.clone())
            else:
                current_parameters.append(param.clone())

        initial_parameters = current_parameters
        generate_corr(current_parameters, initial_parameters, 0, Correlations_vanilla)

        mean_losses = []
        corrects = []
        mean_losses_test = []
        corrects_test = []
        flag = False

        # log weight and grad distribs
        wandb.watch(model_vanilla, log='all')

        for epoch in range(epochs):

            print(f"Epoch {epoch + 1}\n------------------")
            mean_loss, correct = train(train_dl, model_vanilla, loss_fn, optimizer_vanilla, initial_parameters,
                                       Correlations_vanilla, epoch, ALL_par)
            mean_loss_test, correct_test = test(test_dl, model_vanilla, loss_fn)
            #
            mean_losses.append(mean_loss)
            corrects.append(correct)
            mean_losses_test.append(mean_loss_test)
            corrects_test.append(correct_test)
            #
            wandb.log({
                'loss': mean_loss,
                'accuracy': correct,
                'test_loss': mean_loss_test,
                'test_accuracy': correct_test,
                'epoch': epoch + 1
            })
            #
            if correct > 0.95 and flag is False:
                wandb.log({'Threshold': epoch + 1})
                flag = True
        #
        print("Done!")

        #     # In[18]:

        losses_vanilla, accs_vanilla = mean_losses, corrects
        test_losses_vanilla, test_accs_vanilla = mean_losses_test, corrects_test

        print(losses_vanilla, accs_vanilla, test_losses_vanilla, test_accs_vanilla)

        #     # In[19]:

        plt.plot(Correlations_vanilla)

        #     # In[20]:

        vanilla_json = {
            'name': 'vanilla',
            'losses': losses_vanilla,
            'test_losses': test_losses_vanilla,
            'accuracies': accs_vanilla,
            'test_accuracies': test_accs_vanilla,
            'correlations': Correlations_vanilla,
            'lr': lr,
        }

        path = f"finetuning/Adam_{index}_{lr=}.json"

        with open(path, "w") as file:
            json.dump(vanilla_json, file)

        run.finish()

        ## TEST NOISE

        config = {
            "learning_rate": 0.0001,
            "architecture": "resnet18_noise",
            "dataset": f"cifar{CLASSES}",
            "epochs": epochs,
            "lr": lr
        }

        # initiate run
        run = wandb.init(project="ML-project-Correlations",
                         config=config, reinit=True, save_code=True)

        # In[ ]:

        Correlations_noise = []
        ALL_par = True

        current_parameters = []
        for name, param in model_noise.named_parameters():
            if ALL_par:
                if ('conv' in name) or ('fc' in name):
                    current_parameters.append(param.clone())
            else:
                current_parameters.append(param.clone())

        initial_parameters = current_parameters
        generate_corr(current_parameters, initial_parameters, 0, Correlations_noise)

        mean_losses = []
        corrects = []
        mean_losses_test = []
        corrects_test = []
        flag = False

        # log weight and grad distribs
        wandb.watch(model_noise, log='all')

        for epoch in range(epochs):

            print(f"Epoch {epoch + 1}\n------------------")
            mean_loss, correct = train(train_dl, model_noise, loss_fn, optimizer_noise, initial_parameters,
                                       Correlations_noise,
                                       epoch, ALL_par)
            mean_loss_test, correct_test = test(test_dl, model_noise, loss_fn)

            mean_losses.append(mean_loss)
            corrects.append(correct)
            mean_losses_test.append(mean_loss_test)
            corrects_test.append(correct_test)

            wandb.log({
                'loss': mean_loss,
                'accuracy': correct,
                'test_loss': mean_loss_test,
                'test_accuracy': correct_test,
                'epoch': epoch + 1
            })

            if correct > 0.95 and flag is False:
                wandb.log({'Threshold': epoch + 1})
                flag = True

        print("Done!")

        # In[ ]:

        losses_noise, accs_noise = mean_losses, corrects
        test_losses_noise, test_accs_noise = mean_losses_test, corrects_test

        print(losses_noise, accs_noise, test_losses_noise, test_accs_noise)

        # In[ ]:

        plt.plot(Correlations_noise)

        # In[ ]:

        noise_json = {
            'name': 'noise',
            'losses': losses_noise,
            'test_losses': test_losses_noise,
            'accuracies': accs_noise,
            'test_accuracies': test_accs_noise,
            'correlations': Correlations_noise,
            'lr': lr,
        }

        path = f"finetuning/Noise_{index}_{lr=}.json"

        with open(path, "w") as file:
            json.dump(noise_json, file)

        run.finish()

        # ## TEST SWA

        # In[1]:

        config = {
            "learning_rate": 0.0001,
            "architecture": "resnet18_swa",
            "dataset": f"cifar{CLASSES}",
            "epochs": epochs,
            "lr": lr
        }

        # initiate run
        run = wandb.init(project="ML-project-Correlations",
                         config=config, reinit=True, save_code=True)

        # In[ ]:

        Correlations_swa = []
        ALL_par = True  # N.B.: quando ALL_par è True, stiamo loggando solo parte dei parametri e viceversa. Non lo fixo senno ci confondiamo tra esperimenti nuovi e vecchi.

        current_parameters = []
        for name, param in model_swa.named_parameters():
            if ALL_par:
                if ('conv' in name) or ('fc' in name):
                    current_parameters.append(param.clone())
            else:
                current_parameters.append(param.clone())

        initial_parameters = current_parameters
        generate_corr(current_parameters, initial_parameters, 0, Correlations_swa)

        mean_losses = []
        corrects = []
        mean_losses_test = []
        corrects_test = []
        flag = False

        # log weight and grad distribs
        wandb.watch(model_swa, log='all')

        for epoch in range(epochs):

            print(f"Epoch {epoch + 1}  {index=}\n------------------")
            mean_loss, correct = train(train_dl, model_swa, loss_fn, optimizer_swa, initial_parameters,
                                       Correlations_swa,
                                       epoch,
                                       ALL_par)
            mean_loss_test, correct_test = test(test_dl, model_swa, loss_fn)

            mean_losses.append(mean_loss)
            corrects.append(correct)
            mean_losses_test.append(mean_loss_test)
            corrects_test.append(correct_test)

            wandb.log({
                'loss': mean_loss,
                'accuracy': correct,
                'test_loss': mean_loss_test,
                'test_accuracy': correct_test,
                'epoch': epoch + 1
            })

            if correct > 0.95 and flag is False:
                wandb.log({'Threshold': epoch + 1})
                flag = True

        losses_swa, accs_swa = mean_losses, corrects
        test_losses_swa, test_accs_swa = mean_losses_test, corrects_test

        print(losses_swa, accs_swa, test_losses_swa, test_accs_swa)

        plt.plot(Correlations_swa)

        swa_json = {
            "name": "swa",
            'losses': losses_swa,
            'test_losses': test_losses_swa,
            'accuracies': accs_swa,
            'test_accuracies': test_accs_swa,
            'correlations': Correlations_swa,
            'lr': lr,

        }

        path = f"finetuning/SWA_{index}_{lr=}.json"

        with open(path, "w") as file:
            json.dump(swa_json, file)

        run.finish()
