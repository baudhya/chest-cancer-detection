import argparse

class Argument():
    def get_arguments():
        parser = argparse.ArgumentParser(description="Training Configuration")
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
        parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
        parser.add_argument('--num_epoch', type=int, default=10, help='Number of epochs')
        parser.add_argument('--model_name', type=str, default='densenet', help='Model architecture name')
        return parser.parse_args()
