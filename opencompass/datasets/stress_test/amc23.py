import json
import copy
from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset
from .utils import merge_math

@LOAD_DATASET.register_module()
class AMC23Dataset(BaseDataset):

    @staticmethod
    def load(path, **kwargs):
        path = get_data_path(path)
        dataset = []
        with open(path, 'r') as f:
            for line in f:
                line = json.loads(line)
                dataset.append(line)
        return Dataset.from_list(dataset)

@LOAD_DATASET.register_module()
class StressAMC23Dataset(BaseDataset):

    @staticmethod
    def load(path, n_merge: int = 3, **kwargs):
        path = get_data_path(path, local_mode=True)
        dataset = []
        with open(path, 'r') as f:
            for line in f:
                line = json.loads(line)
                dataset.append(line)

        dataset = merge_math(dataset, q_key='question', a_key='answer', n_merge=n_merge)
        return Dataset.from_list(dataset)