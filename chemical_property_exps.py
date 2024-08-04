"""
The training of chemical properties.
"""
import sys
from our_exps.utils import train_type_n_times
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--dataset_name", dest="dataset_name", default='Kraken', type=str, choices= ['Kraken','Drugs','BDE'],
                    required=False)
parser.add_argument("--task", dest="task", default='B5', type=str, choices=['B5','L','burB5','burL','ip','ea','chi'],
                    required=False)
parser.add_argument("--batch_size", dest="batch_size", default=20, type=int, required=False)

args = parser.parse_args()
if args.task == 'Kraken':
  accum_grad = 40 // args.batch_size
else:
  accum_grad = 1

test_acc, val_acc, train_acc = train_type_n_times(types=args.dataset_name, task=args.task, metric_track='acc', fix_seed=False,epochs=1200,batch_size = args.batch_size, accum_grad = accum_grad)

print(f"Train acc {train_acc} in task {args.task}")

print(f"Val acc {val_acc} in task {args.task}")

print(f"Test acc {test_acc} in task {args.task}")
