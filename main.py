
import warnings
warnings.filterwarnings("ignore")

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
    print_classification_report,
    plot_class_frequencies,
)

from utils.config import Configuration
from utils.argument_parser import Argument

#Hyper parameter's 
# IMG_SIZE = 224 # original_image -> 200 x 200 size . Use 224 x 224 for best output (currently not fitting inside my GPU)

class_mapping = {
    'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib': 'Adenocarcinoma',
    'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa': 'Large Cell Carcinoma',
    'normal': 'Normal',
    'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa': 'Squamous Cell Carcinoma'
}


if __name__ == "__main__":
    args = Argument.get_arguments()
    if args.model_name == "custom":
        IMG_SIZE = 224
    else:
        IMG_SIZE = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Getting the dataset
    train_transform, val_transform = get_transformation(IMG_SIZE)
    train_dataset, test_dataset, val_dataset = get_dataset_processed(
        'dataset/train', 'dataset/test', 'dataset/valid',
        train_transform, train_transform, val_transform
    )

    # Mapping the classes to the new names for better understanding
    original_classes = train_dataset.classes
    new_class_names = [class_mapping[old_name] for old_name in train_dataset.classes]
    train_dataset.classes = new_class_names
    test_dataset.classes = new_class_names
    val_dataset.classes = new_class_names
    class_names = train_dataset.classes

    # Getting the dataloader
    train_loader, test_loader, val_loader = get_dataloader(train_dataset, test_dataset, val_dataset)
    plot_class_frequencies(train_loader, val_loader, test_loader, new_class_names)

    
    model = get_model(len(class_names), args.model_name, args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True) # for learning rate decay 
    config = Configuration(model, criterion, optimizer, scheduler, train_loader, val_loader, args.num_epoch, args.lr, device)
    

    trainer = Trainer(config)
    trainer.train()
    test_acc, test_loss = trainer.evaluate(test_loader)

    print(f"Final Test Accuracy: {test_acc:.2f}%")
    plot_epoch_graph(
        args.num_epoch, 
        trainer.get_train_loss_list(), "Train Loss",
        trainer.get_val_loss_list(), "Validation Loss", 
        "Loss",
        args.model_name
    )

    plot_epoch_graph(
        args.num_epoch,
        trainer.get_train_acc_list(), "Train Accuracy",
        trainer.get_val_acc_list(), "Validation Accuracy",
        "Accuracy",
        args.model_name
    )

    y_true, y_pred = get_classification_data(model, test_loader, device)
    print_classification_report(y_true, y_pred, class_names)
    plot_confusion_matrix(y_true, y_pred, class_names, args.model_name)