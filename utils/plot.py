import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"results/confusion_matrix.jpg")
    plt.close()


def get_classification_data(model, test_data_loader, device='cpu'):
    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return y_true, y_pred


def print_classification_report(y_true, y_pred, class_names):
    print("📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


def plot_epoch_graph(num_epochs, plot_first, plot_first_label, plot_second, plot_second_label, yaxis_label):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), plot_first, label=plot_first_label)
    plt.plot(range(1, num_epochs + 1), plot_second, label=plot_second_label)
    plt.xlabel("Epochs")
    plt.ylabel(yaxis_label)
    plt.title(f"{yaxis_label} over Epochs")
    plt.legend()
    plt.savefig(f"results/{yaxis_label}_graph.jpg")