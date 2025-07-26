import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


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

from utils.config import Configuration
from utils.argument_parser import Argument

#Hyper parameter's 
IMG_SIZE = 200 # original_image -> 200 x 200 size . Use 224 x 224 for best output (currently not fitting inside my GPU)


if __name__ == "__main__":
    args = Argument.get_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform, val_transform = get_transformation(IMG_SIZE)
    train_dataset, test_dataset, val_dataset = get_dataset_processed(
        'dataset/train', 'dataset/test', 'dataset/valid',
        train_transform, train_transform, val_transform
    )
    train_loader, test_loader, val_loader = get_dataloader(train_dataset, test_dataset, val_dataset)

    class_names = train_dataset.classes
    model = get_model(len(class_names), args.model_name, True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True) # for learning rate decay 
    config = Configuration(model, criterion, optimizer, scheduler, train_loader, val_loader, args.num_epoch, args.lr, device)
    
    print("Model : ", model)
    print("Classes:", class_names)

    trainer = Trainer(config)
    trainer.train()
    test_acc, test_loss = trainer.evaluate(test_loader)

    print(f"Final Test Accuracy: {test_acc:.2f}%")
    plot_epoch_graph(
        args.num_epoch, 
        trainer.get_train_loss_list(), "Train Loss",
        trainer.get_val_loss_list(), "Validation Loss", 
        "Loss"
    )

    plot_epoch_graph(
        args.num_epoch,
        trainer.get_train_acc_list(), "Train Accuracy",
        trainer.get_val_acc_list(), "Validation Accuracy",
        "Accuracy"
    )

    y_true, y_pred = get_classification_data(model, test_loader, device)
    print_classification_report(y_true, y_pred, class_names)
    plot_confusion_matrix(y_true, y_pred, class_names)