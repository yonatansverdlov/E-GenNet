"""
The training of chemical properties.
"""

from our_code.utils import train_type_n_times

# Can be Drugs, BDE, Kraken.

types = 'Kraken'
# Choose B5 but could be also burL, burB5, L.
task = 'B5'

test_acc, val_acc, train_acc = train_type_n_times(types=types, task=task, metric_track='acc', fix_seed=False)

print(f"Train acc {train_acc} in task {task}")

print(f"Val acc {val_acc} in task {task}")

print(f"Test acc {test_acc} in task {task}")
