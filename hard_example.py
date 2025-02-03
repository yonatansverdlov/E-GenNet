"""
The training of chemical properties.
"""
from our_exps.utils import train_type_n_times


test_acc, val_acc, train_acc = train_type_n_times(types='Hard', task='classify', metric_track='acc', fix_seed=False,epochs=200,batch_size = 1, accum_grad = 1)

print(f"Train acc {train_acc} in task Hard")

print(f"Val acc {val_acc} in task Hard")

print(f"Test acc {test_acc} in task Hard")