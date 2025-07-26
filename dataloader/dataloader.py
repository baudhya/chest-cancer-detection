from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms


def get_dataset_processed(train_path, test_path, val_path, trans_train = None, trans_test = None, trans_val = None):
    if trans_train:
        train_dataset = datasets.ImageFolder(train_path, transform = trans_train)
    else: 
        train_dataset = datasets.ImageFolder(train_path)

    if trans_train:
        test_dataset = datasets.ImageFolder(test_path, transform = trans_test)
    else: 
        test_dataset = datasets.ImageFolder(test_path)
    
    if trans_val:
        val_dataset = datasets.ImageFolder(val_path, transform = trans_val)
    else: 
        val_dataset = datasets.ImageFolder(val_path)

    return (train_dataset, test_dataset, val_dataset)


def get_dataloader(train_dataset, test_dataset, val_dataset, batch_size = 32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (train_loader, test_loader, val_loader)


def get_transformation(img_size = 128):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform