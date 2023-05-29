import torch
from torch import nn

import torchvision
from torchvision import transforms, models

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
EPOCHS = 60
n_checkpoint = 10
for lr in [0.001]:
    for std in [0.01, 0.03, 0.05, 0.75, 0.02, 0.04]:
        model = models.resnet18()
        model.fc = nn.Linear(512, num_classes, bias=True)

        model = model.to(device)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2)
        if wb:
            config = {
                "lr": lr,
                "batch_size": batch_size,
                "model": "Resnet18",
                "pretrained": False,
                "noise": std,
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

        for epoch in range(1, EPOCHS + 1):

            mean_loss, corrects, loss_no_noise, acc_no_noise = utils.train(dl_train, model, loss_fn, optimizer, std=std)

            mean_loss_test, corrects_test = utils.test(dl_val, model, loss_fn)

            if epoch % n_checkpoint == 0 or epoch == 1:
                path = f"models_chkpt/resnet18_adamw_{lr}_{epoch}_noise={std}"
                torch.save(model.state_dict(), path)

            if wb:
                wandb.log({
                    'loss': mean_loss,
                    'accuracy': corrects,
                    'test_loss': mean_loss_test,
                    'test_accuracy': corrects_test,
                    'loss_no_noise': loss_no_noise,
                    'accuracy_no_noise': acc_no_noise,
                    'epoch': epoch
                })

    if wb:
        run.finish()
