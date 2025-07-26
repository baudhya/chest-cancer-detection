from dataclasses import dataclass, field

class Configuration:
    def __init__(self, model, criterion, optimizer, train_dataloader,
        val_dataloader, num_epoch, lr, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_epoch = num_epoch
        self.lr = lr
        self.device = device

    def __repr__(self):
        return (f"Configuration(model={self.model}, criterion={self.criterion}, "
                f"optimizer={self.optimizer}, num_epoch={self.num_epoch}, "
                f"lr={self.lr}, device={self.device})")
