import torch


class Trainer:
    def __init__(self, train_loader, val_loader, model, criterion, optimizer, num_epoch = 10, lr=0.001, device='cpu'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_dataloader = train_loader
        self.val_dataloader  = val_loader
        self.num_epoch = num_epoch
        self.lr = lr
        self.device = device
        self.train_acc_list = []
        self.val_acc_list = []
        self.train_loss_list = []
        self.val_loss_list = []
    
    def train(self):
        for epoch in range(self.num_epoch):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in self.train_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_acc = 100 * correct / total
            val_acc, val_loss = self.evaluate(self.val_dataloader)
            print(f"Epoch [{epoch+1}/{self.num_epoch}] - Train Loss: {running_loss:.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}% - Val Loss: {val_loss:.2f}")
            self.train_acc_list.append(train_acc)
            self.val_acc_list.append(val_acc)
            self.train_loss_list.append(running_loss)
            self.val_loss_list.append(val_loss)


    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_loss /= len(test_loader)
        return 100 * correct / total , val_loss


    def get_train_acc_list(self):
        return self.train_acc_list
    

    def get_val_acc_list(self):
        return self.val_acc_list
    

    def get_train_loss_list(self):
        return self.train_loss_list
    
    def get_val_loss_list(self):
        return self.val_loss_list