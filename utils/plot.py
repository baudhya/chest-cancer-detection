import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np

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



def plot_class_frequencies(train_loader, val_loader, test_loader, class_names):
    total_counts = [0] * len(class_names)
    for loader in (train_loader, val_loader, test_loader):
        for _, labels in loader:
            for label in labels:
                total_counts[label] += 1

    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    bars = plt.bar(class_names, total_counts, color=colors)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height}', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Class Distribution', fontsize=14, pad=20)
    plt.xticks(rotation=45 if len(class_names) > 4 else 0)
    
    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"results/class_distribution.jpg")
    plt.close()

    for class_name, count in zip(class_names, total_counts):
        print(f"{class_name}: {count}")