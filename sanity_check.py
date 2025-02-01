"""
Sanity Checking our_exps.
We show all intermediate and final output are equivariant and invariant.
"""
import os
from easydict import EasyDict
import yaml
from pathlib import Path
<<<<<<< HEAD
from our_exps.model.models import InvariantGenericNet
=======
from model.models import InvariantGenericNet
>>>>>>> Reinitialize egenet repository after moving

types = 'sanity_check'

path = Path(os.path.abspath(__file__)).parent
<<<<<<< HEAD
with open(os.path.join(path, f'our_exps/data/config_files/{types}_config.yaml')) as f:
    type_config = EasyDict(yaml.safe_load(f)[types])
with open(os.path.join(path, f'our_exps/data/config_files/General_config.yaml')) as f:
=======
with open(os.path.join(path, f'data_creation/config_files/{types}_config.yaml')) as f:
    type_config = EasyDict(yaml.safe_load(f)[types])
with open(os.path.join(path, f'data_creation/config_files/General_config.yaml')) as f:
>>>>>>> Reinitialize egenet repository after moving
    general_config = EasyDict(yaml.safe_load(f)['General_config'])
config = EasyDict({'type_config': type_config, 'general_config': general_config, 'type': types, 'task': 'sanity'})

# Random Model.
model = InvariantGenericNet(config=config)
# Random sample.
data_obj = model.return_random_obj()
# Show our model is rotation invariant.
model.check_invariant_rotation(data_obj=data_obj)
# Show our intermediate model is rotation equivariant.
model.check_equivariant_rotation(data_obj=data_obj)
# Show our model is permutation invariant.
model.check_invariant_permutation(data_obj=data_obj)
# Show our intermediate model is permutation equivariant.
model.check_equivariant_permutation(data_obj=data_obj)
