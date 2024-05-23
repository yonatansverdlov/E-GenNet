"""
k_chain experiments.
We first show our model can distinguish 12-chain paths using 7 blocks.
Then, we show we can separate pair A in the paper.
Then, we show we can separate pair B in the paper.
"""
from script.utils import train_type_n_times

num_times = 10

train_k_chain = True

train_pair_A = True

train_pair_B = True

if train_k_chain:
    print("First we train our model to distinguish 12-chain graphs using 7 blocks(minimal)")

    acc = train_type_n_times(types='k_chain', task='classify_original', metric_track='loss', num_times=num_times, fix_seed=True,epochs = 150)

    print(f"The accuracy is {acc[0]}, over {num_times} different seeds")

    print("Succeeding distinguishing the k-chain experiment using 7 blocks")
    input("Press Enter to continue fot Pair A")

if train_pair_A:
    acc = train_type_n_times(types='k_chain', task='classify_pair_A', metric_track='loss', num_times=num_times, fix_seed=True,epochs = 150)

    print(f"The accuracy is {acc[0]}, over {num_times} different seeds")

    print("Succeeding distinguishing Pair A, we now continuing to Pair B")
    input("Press Enter to continue fot Pair B")
if train_pair_B:
    acc = train_type_n_times(types='k_chain', task='classify_pair_B', metric_track='loss', num_times=num_times, fix_seed=True,epochs = 150)

    print(f"The accuracy is {acc[0]}, over {num_times} different seeds")

    print("Succeeding distinguishing Pair B, we are done!")
