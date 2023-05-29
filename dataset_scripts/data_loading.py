from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


def get_data_loaders(train_folder="train.resized", val_folder="val.resized", batch_size=256, num_workers=6):
    t = transforms.Compose([
        transforms.ToTensor()
    ])

    _df_train = ImageFolder(train_folder, transform=t)
    _df_val = ImageFolder(val_folder, transform=t)

    train_loader = DataLoader(
        _df_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)

    val_loader = DataLoader(
        _df_val,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)

    return train_loader, val_loader


if __name__ == '__main__':
    t, v = get_data_loaders()
    z = transforms.ToPILImage()
    for j, ao in t:
        print(ao)
        break
