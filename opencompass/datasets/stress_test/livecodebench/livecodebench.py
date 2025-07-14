# Copyright (c) 2024, LiveCodeBench and its contributors.
# Copyright (c) 2023, OpenCompass and its contributors.

import base64
import json
import pickle
import zlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from datasets import DatasetDict, load_dataset, Dataset, load_from_disk
from opencompass.utils import get_data_path  # noqa: F401, F403
from ..base import BaseDataset
from .prompts import (CodeGenerationPromptConstants,)
from ..utils import merge_livecodebench

class Platform(Enum):
    LEETCODE = 'leetcode'
    CODEFORCES = 'codeforces'
    ATCODER = 'atcoder'


class Difficulty(Enum):
    EASY = 'easy'
    MEDIUM = 'medium'
    HARD = 'hard'


class TestType(Enum):
    STDIN = 'stdin'
    FUNCTIONAL = 'functional'


@dataclass
class Test:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        self.testtype = TestType(self.testtype)


class LCBCodeGenerationDataset(BaseDataset):

    @staticmethod
    def load(path: str = 'opencompass/code_generation_lite',
             local_mode: bool = True,
             release_version: str = 'release_v1',
             start_date: str = None,
             end_date: str = None):

        def transform(item):
            # Define the dataitem mapping logic

            # starter_code
            if item['starter_code']:
                format_prompt = f'### Format: {CodeGenerationPromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n'  # noqa: E501
                format_prompt += f"```python\n{item['starter_code']}\n```\n\n"  # noqa: Q000, E501
            else:
                format_prompt = f'### Format: {CodeGenerationPromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n'  # noqa: E501
                format_prompt += '```python\n# YOUR CODE HERE\n```\n\n'

            item['format_prompt'] = format_prompt

            # load test cases
            public_test_cases = item['public_test_cases']
            public_test_cases = json.loads(item['public_test_cases'])

            private_test_cases = item['private_test_cases']
            try:
                private_test_cases = json.loads(item['private_test_cases'])
            except Exception as e:  # noqa: F841
                private_test_cases = json.loads(
                    pickle.loads(
                        zlib.decompress(
                            base64.b64decode(private_test_cases.encode(
                                'utf-8'))  # type: ignore
                        )))  # type: ignore

            # load metadata
            metadata = json.loads(item['metadata'])
            evaluation_sample = json.dumps({
                'inputs':
                [t['input'] for t in public_test_cases + private_test_cases],
                'outputs':
                [t['output'] for t in public_test_cases + private_test_cases],
                'fn_name':
                metadata.get('func_name', None),
            })
            item['evaluation_sample'] = evaluation_sample

            return item

        path = get_data_path(path, local_mode=local_mode)

        dataset = load_dataset(
            path,  # 'livecodebench/code_generation_lite'
            split='test',
            version_tag=release_version,
            trust_remote_code=True)

        dataset = dataset.map(transform)

        if start_date is not None:
            p_start_date = datetime.strptime(start_date, '%Y-%m-%d')
            dataset = dataset.filter(
                lambda e: p_start_date <= datetime.fromisoformat(e[
                    'contest_date']))  # noqa: E501
        if end_date is not None:
            p_end_date = datetime.strptime(end_date, '%Y-%m-%d')
            dataset = dataset.filter(lambda e: datetime.fromisoformat(e[
                'contest_date']) <= p_end_date)  # noqa: E501

        return DatasetDict({'test': dataset, 'train': dataset})


class StressLCBCodeGenerationDataset(BaseDataset):

    @staticmethod
    def load(path: str = 'opencompass/code_generation_lite',
             local_mode: bool = True,
             release_version: str = 'release_v1',
             start_date: str = None,
             end_date: str = None,
             n_merge: int = 3,
        ):

        def transform(item):
            # Define the dataitem mapping logic

            # starter_code
            if item['starter_code']:
                format_prompt = f'### Format: {CodeGenerationPromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n'  # noqa: E501
                format_prompt += f"```python\n{item['starter_code']}\n```\n\n"  # noqa: Q000, E501
            else:
                format_prompt = f'### Format: {CodeGenerationPromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n'  # noqa: E501
                format_prompt += '```python\n# YOUR CODE HERE\n```\n\n'

            item['format_prompt'] = format_prompt

            # load test cases
            public_test_cases = item['public_test_cases']
            public_test_cases = json.loads(item['public_test_cases'])

            private_test_cases = item['private_test_cases']
            try:
                private_test_cases = json.loads(item['private_test_cases'])
            except Exception as e:  # noqa: F841
                private_test_cases = json.loads(
                    pickle.loads(
                        zlib.decompress(
                            base64.b64decode(private_test_cases.encode(
                                'utf-8'))  # type: ignore
                        )))  # type: ignore

            # load metadata
            metadata = json.loads(item['metadata'])
            evaluation_sample = json.dumps({
                'inputs':
                [t['input'] for t in public_test_cases + private_test_cases],
                'outputs':
                [t['output'] for t in public_test_cases + private_test_cases],
                'fn_name':
                metadata.get('func_name', None),
            })
            item['evaluation_sample'] = evaluation_sample

            return item

        path = get_data_path(path, local_mode=local_mode)
        print(path)
        dataset = load_dataset(
            path,  # 'livecodebench/code_generation_lite'
            split='test',
            version_tag=release_version,
            trust_remote_code=True)
        
        dataset = dataset.map(transform)

        if start_date is not None:
            p_start_date = datetime.strptime(start_date, '%Y-%m-%d')
            dataset = dataset.filter(
                lambda e: p_start_date <= datetime.fromisoformat(e[
                    'contest_date']))  # noqa: E501
        if end_date is not None:
            p_end_date = datetime.strptime(end_date, '%Y-%m-%d')
            dataset = dataset.filter(lambda e: datetime.fromisoformat(e[
                'contest_date']) <= p_end_date)  # noqa: E501
        
        samples = [sample for sample in dataset]
        dataset = merge_livecodebench(samples, q_key="question_content", a_key="question_id", n_merge=n_merge)
        dataset = Dataset.from_list(dataset)

        return DatasetDict({'test': dataset, 'train': dataset})