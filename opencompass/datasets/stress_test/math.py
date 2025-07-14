import json
import os
import re
from os import environ

from datasets import Dataset, DatasetDict

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import (ICL_EVALUATORS, LOAD_DATASET,
                                  TEXT_POSTPROCESSORS)
from opencompass.utils import get_data_path

from .base import BaseDataset
from .utils import merge_math

def last_boxed_only_string(string):
    idx = string.rfind('\\boxed')
    if idx < 0:
        idx = string.rfind('\\fbox')
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == '{':
            num_left_braces_open += 1
        if string[i] == '}':
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def remove_boxed(s):
    left = '\\boxed{'
    try:
        assert s[:len(left)] == left
        assert s[-1] == '}'
        return s[len(left):-1]
    except Exception:
        return None


def extract_boxed_answer(pred_str, strip_double_curly_brace=False):
    boxed_str = last_boxed_only_string(pred_str)
    if boxed_str is None:
        return None
    answer = remove_boxed(boxed_str)
    if answer is None:
        return None
    if strip_double_curly_brace:
        match = re.match('^\{(.*)\}$', answer)  # noqa: W605
        if match:
            answer = match.group(1)
    return answer

@LOAD_DATASET.register_module()
class MATHDataset(BaseDataset):

    @staticmethod
    def load(path: str, file_name: str = 'math.json', **kwargs):
        path = get_data_path(path)
        dataset = DatasetDict()
        raw_data = []
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            ms_dataset = MsDataset.load(path, split='train')
            for item in ms_dataset:
                raw_data.append({
                    'problem':
                    item['problem'],
                    'solution':
                    extract_boxed_answer(item['solution'])
                })
        else:
            file_path = os.path.join(path, file_name)
            data = json.load(open(file_path))
            for i in data.keys():
                raw_data.append({
                    'problem':
                    data[i]['problem'],
                    'solution':
                    extract_boxed_answer(data[i]['solution'])
                })
        dataset['test'] = Dataset.from_list(raw_data)
        dataset['train'] = Dataset.from_list(raw_data)
        return dataset
    
@LOAD_DATASET.register_module()
class StressMATHDataset(BaseDataset):

    @staticmethod
    def load(path: str, file_name: str = 'math.json', n_merge: int = 5, **kwargs):
        path = get_data_path(path, local_mode=True)
        dataset = DatasetDict()
        raw_data = []
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            ms_dataset = MsDataset.load(path, split='train')
            for item in ms_dataset:
                raw_data.append({
                    'question':
                    item['problem'],
                    'answer':
                    extract_boxed_answer(item['solution'])
                })
        else:
            file_path = os.path.join(path, file_name)
            data = json.load(open(file_path))
            for i in data.keys():
                raw_data.append({
                    'question':
                    data[i]['problem'],
                    'answer':
                    extract_boxed_answer(data[i]['solution'])
                })
        raw_data = merge_math(raw_data, q_key='question', a_key='answer', n_merge=n_merge)
        dataset['test'] = Dataset.from_list(raw_data)
        dataset['train'] = Dataset.from_list(raw_data)
        return dataset