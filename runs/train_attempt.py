import torch
from torch import nn

from data_laoding import get_data_loaders
from torchvision.models import resnet18

tdf, vdf = get_data_loaders()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = resnet18()

num_classes = 50
model.fc = nn.Linear(512, num_classes, bias=True)
model.to(device)

#model = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

import time

for epoch in range(10):
    stt = time.time()
    s = stt
    model.train()
    for idx, (image, label) in enumerate(tdf):
        image = image.float()

        image = image.to(device)
        label = label.to(device)

        logits = model(image)
        loss = loss_fn(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 20 == 0:
            c = time.time()
            print(
                f"{idx=} / {len(tdf)} {round(100 * idx / len(tdf), 2)}% elapsed={round(c - s, 2)}s projected epoch={round(len(tdf) / (idx + 1) * (c - stt), 2)}s")
            print(f'loss: {loss}')
            s = time.time()

        # if idx > 2:
        #     break

    del image
    del label

    loss = 0
    correct = 0
    with torch.no_grad():
        model.eval()
        for idx, (image, label) in enumerate(vdf):
            # print(f"test {idx} / {len(vdf)}")
            image = image.float()

            image = image.to(device)
            label = label.to(device)

            logits = model(image)
            loss += loss_fn(logits, label)

            # optimizer.zero_grad()

#            pred = logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            
            correct += (logits.argmax(1) == label).type(torch.float).sum().item()
    
    loss /= len(vdf)
    print(f"Test {loss=} - Accuracy={round(100 * correct / len(vdf.dataset), 2)}%")
