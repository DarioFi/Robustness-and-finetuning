import os.path
import sys

import torch
from torchvision.io import read_image
from torchvision.datasets import ImageFolder
from torchvision import transforms
import PIL

ind = str(sys.argv[1])
# x = read_image("train.X1/n01440764/n01440764_18.JPEG")

# print(x.shape)
# print(type(x))
try:
    FOLDER = str(sys.argv[2])
    fd = FOLDER
except IndexError:
    FOLDER = "train"
    fd = "train.X"

data = ImageFolder(fd + f"{ind}")

t = transforms.PILToTensor()
# print(len(list(data.classes)))
# print(len(list(set(list((data.classes))))))

try:
    os.mkdir(f"{FOLDER}.resized")
except FileExistsError:
    print("resized folder already existing")

cd = set()
for i, (x, y) in enumerate(data):
    if i % 1000 == 0:
        print(f"{i=}/{len(data)}")

    name = f"{FOLDER}.resized/{data.classes[y]}/img{i}.jpg"
    if os.path.isfile(name):
        # print("SKIP")
        continue

    if y not in cd:
        try:
            os.mkdir(f"{FOLDER}.resized/{data.classes[y]}")
            cd.add(y)
        except FileExistsError:
            print("data was preloaded otherwise very sus")
            cd.add(y)

    tr = x.resize((112, 112))
    tr.save(f"{FOLDER}.resized/{data.classes[y]}/img{i}.jpg")

    # tens = t(tr)
    # print(type(tens))
    # print(tens.shape)
    # torch.save(tens, "t1")
    # break


# run like aggregator.py 1
# run like aggregator.py .X val