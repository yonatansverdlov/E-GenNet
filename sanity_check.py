"""
Sanity Checking script.
We show all intermediate and final output are equivariant and invariant.
"""
import os
from easydict import EasyDict
import yaml
from pathlib import Path
from script.model.models import InvariantGenericNet

types = 'sanity_check'

path = Path(os.path.abspath(__file__)).parent
with open(os.path.join(path, f'our_code/data/config_files/{types}_config.yaml')) as f:
    type_config = EasyDict(yaml.safe_load(f)[types])
with open(os.path.join(path, f'our_code/data/config_files/General_config.yaml')) as f:
    general_config = EasyDict(yaml.safe_load(f)['General_config'])
config = EasyDict({'type_config': type_config, 'general_config': general_config, 'type': types, 'task':'sanity'})
model = InvariantGenericNet(config=config)
data_obj = model.return_random_obj()

model.check_invariant_rotation(data_obj=data_obj)
model.check_equivariant_rotation(data_obj=data_obj)
model.check_equivariant_permutation(data_obj=data_obj)
model.check_invariant_permutation(data_obj=data_obj)



