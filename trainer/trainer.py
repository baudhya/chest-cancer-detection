import torch


class Trainer:
    def __init__(self, config):
        self.config = config
        self.train_acc_list = []
        self.val_acc_list = []
        self.train_loss_list = []
        self.val_loss_list = []
    
    def train(self):
        for epoch in range(self.config.num_epoch):
            self.config.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in self.config.train_dataloader:
                inputs, labels = inputs.to(self.config.device), labels.to(self.config.device)

                self.config.optimizer.zero_grad()
                outputs = self.config.model(inputs)
                loss = self.config.criterion(outputs, labels)
                loss.backward()
                self.config.optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_acc = 100 * correct / total
            val_acc, val_loss = self.evaluate(self.config.val_dataloader)
            print(f"Epoch [{epoch+1}/{self.config.num_epoch}] - Train Loss: {running_loss:.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}% - Val Loss: {val_loss:.2f}")
            self.train_acc_list.append(train_acc)
            self.val_acc_list.append(val_acc)
            self.train_loss_list.append(running_loss)
            self.val_loss_list.append(val_loss)


    def evaluate(self, test_loader):
        self.config.model.eval()
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.config.device), labels.to(self.config.device)
                outputs = self.config.model(inputs)
                loss = self.config.criterion(outputs, labels)
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