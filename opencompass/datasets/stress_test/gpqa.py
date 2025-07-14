import csv
import os
import random
import re
import copy
from datasets import Dataset

from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path

from .base import BaseDataset
from .utils import merge_gpqa

@LOAD_DATASET.register_module()
class GPQADataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, **kwargs):
        path = get_data_path(path)
        cnt = 0
        data = []
        with open(os.path.join(path, name), 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if row[7] == 'Question':
                    continue
                cnt = cnt + 1
                question = row[7]
                # 第一个是正确选项
                options = [row[8], row[9], row[10], row[11]]
                shuffle_patterns = ['ABCD', 'BCDA', 'CDAB', 'DABC']  # 更新选项顺序
                c = shuffle_patterns[cnt % 4]
                line = {'question': question}
                ground_truth = options[0]
                for i in range(4):
                    line['ABCD'[i]] = options[ord(c[i]) - ord('A')]
                for i in range(4):
                    if line['ABCD'[i]] == ground_truth:
                        line['answer'] = 'ABCD'[i]
                        break
                data.append(line)
        dataset = Dataset.from_list(data)
        return dataset
    
question_w_options = """
{question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

@LOAD_DATASET.register_module()
class StressGPQADataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, n_merge: int = 4, **kwargs):
        path = get_data_path(path, local_mode=True)
        cnt = 0
        dataset = []
        with open(os.path.join(path, name), 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if row[7] == 'Question':
                    continue
                cnt = cnt + 1
                question = row[7]
                # 第一个是正确选项
                options = [row[8], row[9], row[10], row[11]]
                shuffle_patterns = ['ABCD', 'BCDA', 'CDAB', 'DABC']  # 更新选项顺序
                c = shuffle_patterns[cnt % 4]
                line = {'question': question}
                ground_truth = options[0]
                for i in range(4):
                    line['ABCD'[i]] = options[ord(c[i]) - ord('A')]
                for i in range(4):
                    if line['ABCD'[i]] == ground_truth:
                        line['answer'] = 'ABCD'[i]
                        break
                dataset.append(line)
                
        dataset = merge_gpqa(dataset, q_key='question', a_key='answer', n_merge=n_merge)
                
        dataset = Dataset.from_list(dataset)
        return dataset

@TEXT_POSTPROCESSORS.register_module()
def GPQA_Simple_Eval_postprocess(text: str) -> str:
    ANSWER_PATTERN = r'(?i)ANSWER\s*:\s*([A-D])'
    match = re.search(ANSWER_PATTERN, text)
    if match:
        return match.group(1)
    return None
