import json
import copy
from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset
from .utils import merge_math

@LOAD_DATASET.register_module()
class Aime2025Dataset(BaseDataset):

    @staticmethod
    def load(path, **kwargs):
        path = get_data_path(path)
        dataset = []
        with open(path, 'r') as f:
            for line in f:
                line = json.loads(line)
                question = line['question']
                line['question'] = question[:]
                line['answer'] = line['answer']
                dataset.append(line)
        return Dataset.from_list(dataset)

@LOAD_DATASET.register_module()
class StressAime2025Dataset(BaseDataset):

    @staticmethod
    def load(path, n_merge: int = 3, **kwargs):
        path = get_data_path(path, local_mode=True)
        dataset = []
        with open(path, 'r') as f:
            for line in f:
                line = json.loads(line)
                origin_prompt = line['question']
                line['question'] = origin_prompt[:]
                line['answer'] = line['answer']
                dataset.append(line)

        dataset = merge_math(dataset, q_key='question', a_key='answer', n_merge=n_merge)
        return Dataset.from_list(dataset)