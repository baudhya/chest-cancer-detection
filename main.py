import torch
import torch.nn as nn
import torch.optim as optim

from model.model import get_model
from dataloader.dataloader import (
    get_dataloader, 
    get_dataset_processed, 
    get_transformation,
) 
from trainer.trainer import Trainer
from utils.plot import (
    plot_epoch_graph,
    get_classification_data,
    plot_confusion_matrix,
    print_classification_report
)

#Hyper parameter's
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 200
BATCH_SIZE = 32
lr = 0.0001
NUM_EPOCH = 30
model_name = 'densenet'


train_transform, val_transform = get_transformation(IMG_SIZE)
train_dataset, test_dataset, val_dataset = get_dataset_processed(
    'dataset/train', 'dataset/test', 'dataset/valid',
    train_transform, train_transform, val_transform
)
train_loader, test_loader, val_loader = get_dataloader(train_dataset, test_dataset, val_dataset)


class_names = train_dataset.classes
print("Classes:", class_names)

model = get_model(len(class_names), model_name, True).to(device)
print("Model : ", model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

trainer = Trainer(train_loader, val_loader, model, criterion, optimizer, NUM_EPOCH, lr, device)
trainer.train()
test_acc, test_loss = trainer.evaluate(test_loader)
print(f"Final Test Accuracy: {test_acc:.2f}%")

plot_epoch_graph(
    NUM_EPOCH, 
    trainer.get_train_loss_list(), "Train Loss",
    trainer.get_val_loss_list(), "Validation Loss", 
    "Loss"
)

plot_epoch_graph(
    NUM_EPOCH,
    trainer.get_train_acc_list(), "Train Accuracy",
    trainer.get_val_acc_list(), "Validation Accuracy",
    "Accuracy"
)

y_true, y_pred = get_classification_data(model, test_loader, device)
print_classification_report(y_true, y_pred, class_names)
plot_confusion_matrix(y_true, y_pred, class_names)