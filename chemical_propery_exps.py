"""
The training of chemical properties.
"""
import sys
from exps.utils import train_type_n_times
arg_parse = False
if arg_parse:
    types = str(sys.argv[1])
    task = str(sys.argv[2])
else:
    # Can be Drugs, BDE, Kraken.

    types = 'Drugs'
    # Choose B5 but could be also burL, burB5, L.
    task = 'ip'

test_acc, val_acc, train_acc = train_type_n_times(types=types, task=task, metric_track='acc', fix_seed=False,epochs=1000)

print(f"Train acc {train_acc} in task {task}")

print(f"Val acc {val_acc} in task {task}")

print(f"Test acc {test_acc} in task {task}")
