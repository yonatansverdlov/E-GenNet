"""
The training of chemical properties.
"""
import torch
from utils import train_type_n_times
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--dataset_name", dest="dataset_name", default='Kraken', type=str, choices= ['Kraken','Drugs','BDE'],
                    required=False)
parser.add_argument("--task", dest="task", default='B5', type=str, choices=['B5','L','burB5','burL','ip','ea','chi','BindingEnergy'],
                    required=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu');
props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None;
batch_size = 5 if props and 'A10' in props.name else (20 if props and 'A40' in props.name else 5)
args = parser.parse_args()

if args.dataset_name == 'Kraken':
  accum_grad = 40 // batch_size
else:
  accum_grad = 1

test_acc, val_acc, train_acc = train_type_n_times(types=args.dataset_name, task=args.task, metric_track='acc', fix_seed=False,epochs=1200,batch_size = batch_size, accum_grad = accum_grad)

print(f"Train acc {train_acc} in task {args.task}")

print(f"Val acc {val_acc} in task {args.task}")

print(f"Test acc {test_acc} in task {args.task}")
