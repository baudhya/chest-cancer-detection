import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import numpy as np
from collections import defaultdict

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout() 
    plt.savefig(f"results/confusion_{model_name}_matrix.jpg")
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


def collect_predictions(config, dataloader):
    config.model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)
            outputs = config.model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    return np.concatenate(all_labels), np.concatenate(all_probs)


def print_classification_report(y_true, y_pred, class_names):
    print("📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


def plot_epoch_graph(num_epochs, plot_first, plot_first_label, plot_second, plot_second_label, yaxis_label, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), plot_first, label=plot_first_label)
    plt.plot(range(1, num_epochs + 1), plot_second, label=plot_second_label)
    plt.xlabel("Epochs")
    plt.ylabel(yaxis_label)
    plt.title(f"{yaxis_label} over Epochs")
    plt.legend()
    plt.savefig(f"results/{yaxis_label}_{model_name}_graph.jpg")



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



def plot_multiclass_roc_curve(true_labels, probs, class_names, model_name):
    # Binarize labels
    n_classes = len(class_names)
    binarized_labels = label_binarize(true_labels, classes=range(n_classes))
    
    # Compute ROC/AUC
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(binarized_labels[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'purple',]
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(f"results/roc_{model_name}_graph.jpg")
    plt.show()




def save_unique_class_row(dataloader, class_names, num_classes=4, samples_per_class=4):
    class_samples = defaultdict(list)
    completed_classes = set()

    # Iterate through the dataloader to find samples
    for images, labels in dataloader:
        for i in range(len(images)):
            label = labels[i].item()
            if label not in completed_classes and len(class_samples[label]) < samples_per_class:
                class_samples[label].append(images[i])
                if len(class_samples[label]) == samples_per_class:
                    completed_classes.add(label)
            
            # Stop once we have enough completed classes
            if len(completed_classes) >= num_classes:
                break
        if len(completed_classes) >= num_classes:
            break
    
    if len(completed_classes) < num_classes:
        print(f"Warning: Could only find {len(completed_classes)} classes with {samples_per_class} samples each.")
        return

    selected_class_indices = sorted(list(completed_classes))[:num_classes]

    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(12, 9))
    fig.suptitle(f'{samples_per_class} Samples From {num_classes} Different Classes', fontsize=16)

    for row_idx, class_idx in enumerate(selected_class_indices):
        images_for_class = class_samples[class_idx]
        axes[row_idx, 0].set_ylabel(class_names[class_idx], rotation=90, size='large')
        
        for col_idx, image in enumerate(images_for_class):
            ax = axes[row_idx, col_idx]
            img_for_display = image.numpy().transpose((1, 2, 0))
            img_for_display = np.clip(img_for_display, 0, 1)

            ax.imshow(img_for_display)
            ax.axis('off')    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"results/class_samples_grid.png", dpi=300)
    plt.close(fig)