#!/usr/bin/env python
# coding: utf-8

# In[84]:


import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms, models

import numpy as np
import matplotlib.pyplot as plt

from utils import *

import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# In[86]:


from torchvision.models import resnet18
from torch.optim.swa_utils import AveragedModel

for model_name, name1 in [("resnet_swa", "SWA"), ("resnet_vanilla", "Adam"), ("resnet_noise", "Noise")]:

    model1 = resnet18(num_classes=50)
    sd = torch.load("good_pretrained_models/" + model_name)
    if "swa" in model_name:
        #     print(sd.keys())
        ks = list(sd.keys())
        for key in ks:
            if "module." not in key:
                continue

            nk = key.replace("module.", "")
            sd[nk] = sd[key]
            del sd[key]
        del sd["n_averaged"]

    model1.load_state_dict(sd)
    loss_fn = nn.CrossEntropyLoss()
    model1 = model1.to(device)

    # In[87]:

    model_name = "resnet_vanilla"
    name2 = "Adam"

    model2 = resnet18(num_classes=50)
    sd = torch.load("good_pretrained_models/" + model_name)
    if "swa" in model_name:
        #     print(sd.keys())
        ks = list(sd.keys())
        for key in ks:
            if "module." not in key:
                continue

            nk = key.replace("module.", "")
            sd[nk] = sd[key]
            del sd[key]
        del sd["n_averaged"]

    model2.load_state_dict(sd)
    loss_fn = nn.CrossEntropyLoss()
    model2 = model2.to(device)

    # In[88]:

    import data_loading

    dl_train, dl_val = data_loading.get_data_loaders()

    # In[89]:

    dl, test_dl = data_loading.get_data_loaders(train_folder="Stratified_sample", batch_size=64)

    do_traverse = False
    do_fsgm = False
    do_corrupted = False
    do_flatness = True
    do_masking = False

    # # Traverse loss landscape along a straight line

    if do_traverse:
        print('Training data:')
        test(dl, model1, loss_fn), test(dl, model2, loss_fn)

        # In[96]:

        print('Test data:')
        test(test_dl, model1, loss_fn), test(test_dl, model2, loss_fn)

        # In[97]:

        nsteps = 50

        lst1, l2_distance, l2_convs, l2_bn = traverse(dl, model1, model2, loss_fn, nsteps, chkpt='chkpt.pt')
        lst2, _, _, _ = traverse(test_dl, model1, model2, loss_fn, nsteps, chkpt='chkpt.pt')

        # In[98]:

        losses = [i[0] for i in lst1]
        test_losses = [i[0] for i in lst2]

        # In[99]:

        print((l2_convs))

        # In[100]:

        accs = [i[1] for i in lst1]
        test_accs = [i[1] for i in lst2]

        # In[101]:

        traverse_json = {
            'nsteps': nsteps,
            'losses': losses,
            'test_losses': test_losses,
            'accuracies': accs,
            'test_accuracies': test_accs,
            'l2_distance': float(l2_distance),
            'l2_convs': float(l2_convs),
            'l2_bn': float(l2_bn),
            'model1': name1,
            'model2': name2
        }

        path = f"Logged robustness/traverse_{name1}_{name2}.json"

        with open(path, "w") as file:
            json.dump(traverse_json, file)

        # In[102]:

        x_axis = np.linspace(0, 1, nsteps + 1)

        plt.scatter([0, 1], [losses[0], losses[-1]])
        plt.scatter([0, 1], [test_losses[0], test_losses[-1]])
        plt.plot(x_axis, losses, label='train')
        plt.plot(x_axis, test_losses, label='test')
        plt.title("Cross-Entropy loss along the line joining the models")
        plt.xlabel("convex combination parameter")
        plt.ylabel("cross entropy loss")
        plt.legend()
        plt.savefig(f"Figure/Traverse {name1}-{name2} loss")
        plt.show()

        # In[103]:

        x_axis = np.linspace(0, 1, nsteps + 1)

        plt.scatter([0, 1], [accs[0], accs[-1]])
        plt.scatter([0, 1], [test_accs[0], test_accs[-1]])
        plt.plot(x_axis, accs, label='train')
        plt.plot(x_axis, test_accs, label='test')
        plt.xlabel("convex combination parameter")
        plt.ylim(0, 1)
        plt.ylabel("accuracy")
        plt.axhline(0.02, color="grey", label="random")
        plt.title("Accuracy loss along the line joining the models")
        plt.legend()
        plt.savefig(f"Figure/Traverse {name1}-{name2} accuracy")
        plt.show()

    # ## Fast Gradient Signed Method attack

    # In[104]:
    if do_fsgm:

        # epss = [0.0, 0.001, 0.005, 0.01, 0.05]

        epss = [x / 1000 for x in range(0, 31)]
        print(epss)
        # lists are better for ordering purposes + matplotlib complains
        losses, test_losses, accs, test_accs = [], [], [], []
        max_b = None

        # epss = [0.01]
        for eps in epss:
            print(eps)
            acc, loss = test_fgsm(dl, model1, eps, max_b, loss_fn)
            test_acc, test_loss = test_fgsm(test_dl, model1, eps, max_b, loss_fn)

            losses.append(loss)
            test_losses.append(test_loss)
            accs.append(acc)
            test_accs.append(test_acc)

        # In[105]:

        losses = [float(x) for x in losses]
        test_losses = [float(x) for x in test_losses]

        # In[106]:

        fgsm_json = {
            'epss': epss,
            'losses': losses,
            'test_losses': test_losses,
            'accuracies': accs,
            'test_accuracies': test_accs,
            'model': name1,
        }

        path = f"Logged robustness/fgsm_{name1}.json"

        with open(path, "w") as file:
            json.dump(fgsm_json, file)

        # In[110]:

        plt.plot(epss, losses, label='train')
        plt.plot(epss, test_losses, label='test')
        plt.xlabel("Attack strength (epsilon)")
        plt.ylabel("CrossEntropy")
        plt.title("Robustness to adversarial attacks - loss")
        plt.legend()
        plt.grid()
        plt.savefig(f"Figure/FGSM {name1} loss")
        plt.show()

        # In[111]:

        plt.plot(epss, accs, label='train')
        plt.plot(epss, test_accs, label='test')
        plt.title("Robustness to adversarial attacks - accuracy")
        plt.xlabel("Attack strength (epsilon)")
        plt.ylabel("Accuracy")

        plt.axhline(0.02, color="grey", label="random")
        plt.legend()
        plt.grid()

        plt.savefig(f"Figure/FGSM {name1} accuracy")
        plt.show()

    # # Classification with corrupted images

    # In[112]:
    if do_corrupted:

        sigmas = [0.0, 0.05, .1, .15, .2, .25, .3, .4, .5, .8]
        sigmas = [x / 100 for x in range(0, 101, 5)]
        max_b = None
        iters = 10

        losses, accs = denoising(model1, dl, loss_fn, sigmas, iters=iters, max_b=max_b)
        test_losses, test_accs = denoising(model1, test_dl, loss_fn, sigmas, iters=iters, max_b=max_b)

        # for sigma in sigmas:
        #     losses[sigma] = sum(losses[sigma]) / len(losses[sigma])
        #     test_losses[sigma] = sum(test_losses[sigma]) / len(test_losses[sigma])

        #     accs[sigma] = sum(accs[sigma]) / len(accs[sigma])
        #     test_accs[sigma] = sum(test_accs[sigma]) / len(test_accs[sigma])

        # In[113]:

        losses

        # In[114]:

        losses_list = []
        test_losses_list = []
        accs_list = []
        test_accs_list = []

        for k in sigmas:
            losses_list.append(float(losses[k]))
            test_losses_list.append(float(test_losses[k]))
            accs_list.append(accs[k])
            test_accs_list.append(test_accs[k])

        losses = losses_list
        test_losses = test_losses_list
        accs = accs_list
        test_accs = test_accs_list

        # In[115]:

        corrupted_json = {
            'sigmas': sigmas,
            'iters': iters,
            'losses': losses,
            'test_losses': test_losses,
            'accuracies': accs,
            'test_accuracies': test_accs,
            'model': name1,
        }

        path = f"Logged robustness/corrupted_{name1}.json"

        with open(path, "w") as file:
            json.dump(corrupted_json, file)

        # In[116]:

        plt.plot(sigmas, losses, label='train')
        plt.plot(sigmas, test_losses, label='test')
        plt.title('Robustness to gaussian noise - Loss')
        plt.xlabel("std of additive noise")
        plt.ylabel("cross entropy loss")
        plt.legend()
        plt.grid()
        plt.savefig(f"Figure/Corrupted {name1} loss")
        plt.show()

        # In[117]:

        plt.plot(sigmas, accs, label='train')
        plt.plot(sigmas, test_accs, label='test')
        plt.title('Robustness to gaussian noise - Accuracy')
        plt.xlabel("std of additive noise")
        plt.ylabel("accuracy")
        plt.legend()
        plt.grid()
        plt.axhline(0.02, color="grey", label="random")
        plt.savefig(f"Figure/Corrupted {name1} accuracy")
        plt.show()

    # # Flatness

    if do_flatness:

        # In[119]:
        # sigmas = [0.0, 0.05, 0.1, 0.15, .2, 0.25, .3, .35, .4, .5, .6, .7, .8, .9]
        sigmas = [x / 100 for x in range(0, 100, 5)]

        # sigmas = [0, 0.5]
        iters = 25
        max_b = None
        losses, accs = weight_flatness(model1, dl, loss_fn, sigmas, iters=iters, chkpt='checkpoint.pt', max_b=max_b)
        test_losses, test_accs = weight_flatness(model1, test_dl, loss_fn, sigmas, iters=iters, chkpt='checkpoint.pt',
                                                 max_b=max_b)

        for sigma in sigmas:
            losses[sigma] = sum(losses[sigma]) / len(losses[sigma])
            test_losses[sigma] = sum(test_losses[sigma]) / len(test_losses[sigma])

            accs[sigma] = sum(accs[sigma]) / len(accs[sigma])
            test_accs[sigma] = sum(test_accs[sigma]) / len(test_accs[sigma])

        # In[120]:

        losses_list = []
        test_losses_list = []
        accs_list = []
        test_accs_list = []

        for k in sigmas:
            losses_list.append(float(losses[k]))
            test_losses_list.append(float(test_losses[k]))
            accs_list.append(accs[k])
            test_accs_list.append(test_accs[k])

        losses = losses_list
        test_losses = test_losses_list
        accs = accs_list
        test_accs = test_accs_list

        # In[121]:

        losses

        # In[122]:

        flat_json = {
            'sigmas': sigmas,
            'iters': iters,
            'losses': losses,
            'test_losses': test_losses,
            'accuracies': accs,
            'test_accuracies': test_accs,
            'model': name1,
        }

        path = f"Logged robustness/flat_{name1}.json"

        with open(path, "w") as file:
            json.dump(flat_json, file)

        # In[123]:

        plt.plot(sigmas, losses, label='train')
        plt.plot(sigmas, test_losses, label='test')
        plt.title('Flatness in weight space - Loss')
        plt.xlabel("std of multiplicative noise")
        plt.ylabel("cross entropy loss")
        plt.legend()
        plt.grid()
        plt.savefig(f"Figure/Flatness {name1} loss")
        plt.show()

        # In[124]:

        plt.plot(sigmas, accs, label='train')
        plt.plot(sigmas, test_accs, label='test')
        plt.title('Flatness in weight space - Accuracy')
        plt.xlabel("std of multiplicative noise")
        plt.ylabel("accuracy")
        plt.axhline(0.02, color="grey", label="random")
        plt.grid()

        plt.legend()
        plt.savefig(f"Figure/Flatness {name1} accuracy")
        plt.show()

    # # Masking

    if do_masking:

        # In[125]:

        # model1 = models.resnet18(weights='IMAGENET1K_V1')
        # model1.fc = nn.Linear(512, num_classes, bias=True)

        # model2 = models.resnet18(weights='IMAGENET1K_V1')
        # model2.fc = nn.Linear(512, num_classes, bias=True)

        # model1.to(device)
        # model2.to(device)

        # loss_fn = torch.nn.CrossEntropyLoss()

        # optimizer1 = torch.optim.AdamW(model1.parameters())

        # In[126]:

        def mask(dataloader, model, loss_fn, max_b=None, dx=10, dy=10, black=None):
            if black is None:
                black = 0

            if max_b is None:
                max_b = len(dataloader)

            mean_loss = 0
            correct = 0
            size = 0

            for idx, (img, label) in enumerate(dataloader):
                B, C, W, H = img.shape
                x, y = np.random.randint(W - dx), np.random.randint(H - dy)

                img[:, :, x: x + dx, y: y + dy] = black
                img, label = img.to(device), label.to(device)

                pred = model(img)
                loss = loss_fn(pred, label)
                mean_loss += loss.item()
                correct += (pred.argmax(1) == label).type(torch.float).sum().item()

                size += len(img)

                if idx + 1 >= max_b:
                    break

            return mean_loss / max_b, correct / size


        # In[129]:

        import copy

        sizes = [x for x in range(0, 100, 5)]
        dxs, dys = copy.copy(sizes), copy.copy(sizes)

        max_b = None
        iters = 10

        losses, accs = [], []
        test_losses, test_accs = [], []

        for dx, dy in zip(dxs, dys):

            print(dx, dy)
            loss, acc, test_loss, test_acc = 0, 0, 0, 0

            for i in range(iters):
                #         print(i)
                loss_temp, acc_temp = mask(dl, model1, loss_fn, max_b=max_b, dx=dx, dy=dy, black=None)
                test_loss_temp, test_acc_temp = mask(test_dl, model1, loss_fn, max_b=max_b, dx=dx, dy=dy, black=None)

                loss += loss_temp
                acc += acc_temp
                test_loss += test_loss_temp
                test_acc += test_acc_temp

            loss /= iters
            test_loss /= iters
            acc /= iters
            test_acc /= iters

            losses.append(loss)
            accs.append(acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)

        # In[130]:

        mask_json = {
            'dxs': dxs,
            'dys': dys,
            'iters': iters,
            'losses': losses,
            'test_losses': test_losses,
            'accuracies': accs,
            'test_accuracies': test_accs,
            'model': name1,
        }

        path = f"Logged robustness/mask_{name1}.json"

        with open(path, "w") as file:
            json.dump(mask_json, file)

        # In[131]:

        plt.plot(dxs, losses, label='train')
        plt.plot(dxs, test_losses, label='test')
        plt.title('Masking of images - Loss')

        plt.legend()
        plt.grid()
        plt.savefig(f"Figure/masking {name1} loss")
        plt.show()

        # In[132]:

        plt.plot(dxs, accs, label='train')
        plt.plot(dxs, test_accs, label='test')
        plt.title('Masking of images - Accuracy')
        plt.axhline(0.02, color="grey", label="random")

        plt.legend()
        plt.grid()
        plt.ylim(0, 1)

        plt.savefig(f"Figure/masking {name1} accuracy")
        plt.show()
