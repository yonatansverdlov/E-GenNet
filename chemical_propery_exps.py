"""
The training of chemical properties.
"""
import sys
import argparse
from exps.utils import train_type_n_times

parser = argparse.ArgumentParser(description='QM9 Example')
parser.add_argument('--dataset_name', type=str, default='Kraken',
                    help='experiment_name')
parser.add_argument('--task', type=str, default='B5',
                    help='input batch size for training (default: 128)')
parser = parser.parse_args()


types = parser.dataset_name
# Choose B5 but could be also burL, burB5, L.
task = parser.task

test_acc, val_acc, train_acc = train_type_n_times(types=types, task=task, metric_track='acc')

print(f"Train acc {train_acc} in task {task}")

print(f"Val acc {val_acc} in task {task}")

print(f"Test acc {test_acc} in task {task}")
