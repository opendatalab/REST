import json
import os
import re
from os import environ

from datasets import Dataset, DatasetDict

from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path

from .base import BaseDataset
from .utils import merge_math

@LOAD_DATASET.register_module()
class GSM8KDataset(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path)
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            dataset = MsDataset.load(dataset_name=path)
        else:
            datasets = {}
            for split in ['train', 'test']:
                split_path = os.path.join(path, split + '.jsonl')
                dataset = []
                with open(split_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = json.loads(line.strip())
                        dataset.append(line)
                datasets[split] = Dataset.from_list(dataset)
            dataset = DatasetDict(datasets)
        return dataset

@LOAD_DATASET.register_module()
class StressGSM8KDataset(BaseDataset):

    @staticmethod
    def load(path, n_merge: int = 5):
        path = get_data_path(path, local_mode=True)
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            # from modelscope import MsDataset
            # dataset = MsDataset.load(dataset_name=path)
            raise NotImplementedError("ModelScope does not support stress test")
        else:
            datasets = {}
            for split in ['train', 'test']:
                split_path = os.path.join(path, split + '.jsonl')
                dataset = []
                with open(split_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = json.loads(line.strip())
                        dataset.append(line)
                        line['answer'] = gsm8k_dataset_postprocess(line['answer'])
                dataset = merge_math(dataset, q_key='question', a_key='answer', n_merge=n_merge)
                datasets[split] = Dataset.from_list(dataset)
            dataset = DatasetDict(datasets)
        return dataset
    

@TEXT_POSTPROCESSORS.register_module('gsm8k_dataset')
def gsm8k_dataset_postprocess(text: str) -> str:
    return text.split('#### ')[1].replace(',', '')


@TEXT_POSTPROCESSORS.register_module('gsm8k')
def gsm8k_postprocess(text: str) -> str:
    text = text.split('Question:')[0]
    numbers = re.findall(r'\-?\d+\.\d+|\-?\d+', text)
    if not numbers:
        return 'NULL'
    return numbers[-1]
