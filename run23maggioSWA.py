import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms, models
from torchvision.models import efficientnet_b0, efficientnet_v2_s

from torch.optim.swa_utils import AveragedModel, SWALR

import numpy as np
import matplotlib.pyplot as plt

import math
import utils
import wandb
import data_loading

wb = True
project = "AI project exam"

if wb:
    wandb.login(key="aaaa6a0838d0e4f2d3f68ffbc76dcb40602660cf", relogin=True, force=False)  # mattia

device = utils.device

batch_size = 256

dl_train, dl_val = data_loading.get_data_loaders(batch_size=batch_size)

l2 = 0.01
num_classes = 50
log_interval = 10
EPOCHS = 40
n_checkpoint = 10
for lr in [0.001, ]:
    for swa_lr in [0.005, 0.001, 0.003]:

        model = models.resnet18()
        model.fc = nn.Linear(512, num_classes, bias=True)

        model = model.to(device)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2)

        swa_model = AveragedModel(model)
        swa_model = swa_model.to(device)
        swa_threshold = .95

#        strat = "linear"
#        anneal_epochs = 5
#        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr, anneal_strategy=strat, anneal_epochs=anneal_epochs)

        if wb:
            config = {
                "lr": lr,
  #              "swa_lr": swa_lr,
                "swa_threshold": swa_threshold,
               # "anneal_strategy": strat,
               # "anneal_epochs": anneal_epochs,
                "batch_size": batch_size,
                "model": "Resnet18",
                "pretrained": False,
                "epochs": EPOCHS,
                "l2": l2,
            }

            run = wandb.init(
                project=project,
                config=config,
                reinit=True,
                save_code=True,
            )

            wandb.watch(
                model,
                log_freq=log_interval,
                log="all"
            )

        swa_started_flag = False
        for epoch in range(1, EPOCHS + 1):

            mean_loss, corrects = utils.train(dl_train, model, loss_fn, optimizer)

            mean_loss_test, corrects_test = utils.test(dl_val, model, loss_fn)

            if corrects > swa_threshold or swa_started_flag:
                swa_started_flag = True
                swa_model.update_parameters(model)
 #               swa_scheduler.step()

            if epoch % n_checkpoint == 0:
                path = f"models_chkpt/resnet18_adamw_{lr=}_{swa_lr=}_{epoch=}"
                torch.save(model.state_dict(), path)

            if wb:
                wandb.log({
                    'loss': mean_loss,
                    'accuracy': corrects,
                    'test_loss': mean_loss_test,
                    'test_accuracy': corrects_test,
                    'epoch': epoch
                })
        swa_model = swa_model.to(device) # superfluo

        trainer = ((x.to(device), y.to(device)) for x,y in dl_train)

        torch.optim.swa_utils.update_bn(trainer, swa_model)
        
        swa_model = swa_model.to(device) # superfluo
        swa_loss, swa_accuracy = utils.test(dl_val, swa_model, loss_fn)

        if wb:
            wandb.log({
                'swa_loss': swa_loss,
                'swa_accuracy': swa_accuracy,
            })

        if wb:
            run.finish()

        swa_path = f"models_chkpt/resnet18_adamw_swa_{lr}_{swa_lr}_{epoch=}"
        torch.save(swa_model.state_dict(), swa_path)

        break # this is added because the swa_lr is irrelevant now
