import os
import os.path as osp
from copy import deepcopy
from typing import Dict, List, Optional

import mmengine
from datasets import Dataset
from mmengine.config import ConfigDict

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.registry import (DICT_POSTPROCESSORS, ICL_PROMPT_TEMPLATES,
                                  TEXT_POSTPROCESSORS)
from opencompass.utils import build_dataset_from_cfg, build_model_from_cfg
from opencompass.utils.logging import get_logger
from opencompass.utils.text_postprocessors import extract_non_reasoning_content_and_apply_template, split_boxed_content

logger = get_logger(__name__)


class StressTestLLMExtractEvaluator(BaseEvaluator):
    """Generic LLM evaluator.

    Arguments:
        prompt_template (ConfigDict): The prompt template for evaluation.
        judge_cfg (ConfigDict): The config for Judge LLM.
        dataset_cfg (ConfigDict): The config for dataset.
        pred_postprocessor (ConfigDict): The config for postprocessor.
            used for the prediction results.
        dict_postprocessor (ConfigDict): The config for postprocessor,
            used for evaluation results dict.
    """

    def __init__(
        self,
        prompt_template: ConfigDict,
        judge_cfg: ConfigDict,
        dataset_cfg: Optional[ConfigDict] = None,
        pred_postprocessor: Optional[ConfigDict] = None,
        dict_postprocessor: Optional[ConfigDict] = None,
        keep_predictions: bool = False,
    ) -> None:
        super().__init__(pred_postprocessor=pred_postprocessor)
        # If judge_cfg is not provided, fall back to the default configuration
        if not judge_cfg:
            self.judge_cfg = self.default_judge_cfg
        else:
            self.judge_cfg = judge_cfg
        self.output_path = ''

        self.prompt_template = ICL_PROMPT_TEMPLATES.build(prompt_template)

        # Build Dataset
        self.dataset_cfg = dataset_cfg
        assert dataset_cfg is not None, 'dataset_cfg is None'

        self.dict_postprocessor = dict_postprocessor
        self.pred_postprocessor = pred_postprocessor

    def build_inferencer(self):
        """Build LLM Inference."""

        self.output_path = f'{self._out_dir}_replica{self.dataset_replica_idx}.json'  # noqa
        logger.info(f'LLM judge details will be saved at:{self.output_path}')
        out_dir, out_name = osp.split(self.output_path)

        logger.info(
            f'Set self.output_path to {self.output_path} for current task')
        assert self.output_path is not None, 'output_path is None'

        # Build LLM Inference
        max_out_len = self.judge_cfg.get('max_out_len', None)
        batch_size = self.judge_cfg.get('batch_size', None)

        model = build_model_from_cfg(model_cfg=self.judge_cfg)

        self.inferencer = GenInferencer(
            model,
            max_out_len=max_out_len,
            batch_size=batch_size,
            output_json_filepath=out_dir,
            output_json_filename=out_name,
        )


    def _score_rule(self, predictions, references, test_set=None):
        try:
            from latex2sympy2_extended import NormalizationConfig
            from math_verify import (ExprExtractionConfig,
                                     LatexExtractionConfig, parse, verify)
        except ImportError:
            raise ImportError('Failed to import required modules. Please '
                              'install the necessary packages: '
                              'pip install math_verify latex2sympy2_extended')
        if isinstance(predictions, list):
            ori_predictions = deepcopy(predictions)
            predictions = [pred for preds in predictions for pred in preds]
        if isinstance(references, list):
            ori_references = deepcopy(references)
            references = [ref for refs in references for ref in refs]
        if len(predictions) > len(references):
            predictions = predictions[:len(references)]
        elif len(predictions) < len(references):
            predictions = predictions + ["X" for _ in range(len(references) - len(predictions))]
        if test_set is not None and "question_list" in test_set[0]:
            ori_questions = deepcopy([item["question_list"] for item in test_set])
            questions = [q for qs in ori_questions for q in qs]
        elif test_set is not None and "problem_list" in test_set[0]:
            ori_questions = deepcopy([item["problem_list"] for item in test_set])
            questions = [q for qs in ori_questions for q in qs]
        else:
            ori_questions = None
            questions = None
        assert len(predictions) == len(
            references), 'predictions and references must have the same length'
        correct = 0
        total_count = 0
        splited_details = []
        for idx, (i, j) in enumerate(zip(predictions, references)):
            total_count += 1
            j_with_env = f'${j}$'
            gold_parsed = parse(
                j_with_env,
                extraction_mode='first_match',
                extraction_config=[
                    LatexExtractionConfig(),
                    ExprExtractionConfig(),
                ],
            )

            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct
                # latex (no malformed operators)
                answer_parsed = parse(
                    i,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed='all',
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode='first_match',
                )

                answer_correct = float(verify(answer_parsed, gold_parsed))
                correct += answer_correct
                detail = {
                    'prediction': str(answer_parsed),
                    'reference': str(gold_parsed),
                    'correct': True if answer_correct else False,
                }
                if questions is not None:
                    detail['question'] = questions[idx]
                splited_details.append(detail)
        cnt = 0
        merged_details = []
        for idx, (preds, refs) in enumerate(zip(ori_predictions, ori_references)):
            assert len(preds) == len(refs), f"predictions and references have different length: {len(preds)} != {len(refs)}"
            if ori_questions is not None:
                qs = ori_questions[idx]
                assert len(qs) == len(preds), f"questions and predictions have different length: {len(qs)} != {len(preds)}"
            correct_list = []
            for i in range(len(preds)):
                correct_list.append(splited_details[cnt]['correct'])
                cnt += 1
            detail = {
                "prediction": str(preds),
                "reference": str(refs),
                "correct": str(correct_list[:]),
            }
            if ori_questions is not None:
                detail['question'] = str(qs)
            merged_details.append(detail)
        result = {'accuracy': 100 * correct / total_count, 'merged_details': merged_details, 'splited_details': splited_details}
        return result
    
    def score(
        self,
        predictions,
        references: Optional[List] = None,
        test_set: Optional[Dataset] = None,
    ) -> Dict:
        """Apply to single-model scoring.

        Args:
            predictions: List of model predictions
            references: List of reference answers
            test_set: Optional Dataset containing additional
            context for evaluation
        """
        assert len(predictions) == len(
            references), 'predictions and references must have the same length'

        # -------------- Build Inferencer ----------------
        self.build_inferencer()
        # ---------------- Process Predictions ------------------
        predictions = self.pred_postprocess(predictions)

        # # For Single Round Dialogue
        # prediction_dict = {'prediction': predictions, 'obj_gold': references}
        # For Single Round Dialogue
        prediction_dict = {'prediction': predictions}

        # ---------------- Build Dataset for LLM Judge -----------------
        if self.dataset_cfg:
            dataset = build_dataset_from_cfg(self.dataset_cfg)
            for k, v in prediction_dict.items():
                dataset.reader.dataset['test'] = dataset.test.add_column(k, v)
                dataset.reader.input_columns.append(k)

            # if references:
            #     dataset.reader.input_columns.append('reference')
            #     dataset.reader.dataset['test'] = dataset.test.add_column(
            #         'reference', references)
        else:
            # Handle test_set in the else branch
            from opencompass.datasets.lmeval import LMEvalDataset

            if test_set is not None:
                # If test_set is provided, use it as the base
                # Ensure necessary columns exist
                if 'prediction' not in test_set.column_names:
                    test_set = test_set.add_column('prediction', predictions)
                if 'reference' not in test_set.column_names:
                    test_set = test_set.add_column('reference', references)

                # Prepare input_columns and data dictionary
                input_columns = test_set.column_names
                data_dict = {
                    column: test_set[column]
                    for column in test_set.column_names
                }
            else:
                # Original default dataset building logic
                input_columns = list(prediction_dict.keys())
                if references:
                    input_columns.append('reference')
                data_dict = prediction_dict.copy()
                if references:
                    data_dict['reference'] = references

            # Create LMEvalDataset
            dataset = LMEvalDataset(
                reader_cfg=dict(
                    input_columns=input_columns,
                    output_column=None,
                    train_split='test',
                ),
                **data_dict,
            )

        # dataset.reader.output_column = 'prediction'
        retriever = ZeroRetriever(dataset)
        # ----------------- LLM Judge ----------------
        self.inferencer.inference(retriever=retriever,
                                  prompt_template=self.prompt_template)

        output = mmengine.load(self.output_path)
        # return self.output_postprocess(output, dataset)
        predictions, references = [], []
        for k, v in output.items():
            predictions.append(split_boxed_content(v['prediction'], n_merge=self.dataset_cfg.n_merge))
            references.append(split_boxed_content(v['gold'], n_merge=self.dataset_cfg.n_merge))

        results = self._score_rule(predictions, references)
        return results

    def pred_postprocess(self, predictions: List) -> Dict:
        if self.pred_postprocessor is None:
            return predictions
        else:
            kwargs = self.pred_postprocessor
            proc = TEXT_POSTPROCESSORS.get(kwargs.pop('type'))
            return [proc(pred, **kwargs) for pred in predictions]

    def output_postprocess(self, output: Dict, dataset=None) -> Dict:
        """Postprocess output by adding necessary statistics or data into
        it."""
        import inspect

        if self.dict_postprocessor is None:
            return output
        else:
            kwargs = deepcopy(self.dict_postprocessor)
            proc = DICT_POSTPROCESSORS.get(kwargs.pop('type'))
            sig = inspect.signature(proc)
            if 'dataset' in sig.parameters:
                return proc(output,
                            self.output_path,
                            dataset=dataset,
                            **kwargs)
            else:
                return proc(output, self.output_path, **kwargs)

    @property
    def default_judge_cfg(self):
        from opencompass.models import OpenAISDK
        logger.info('Please set your judge model in `OC_JUDGE_MODEL`, \
            `OC_JUDGE_API_KEY`, `OC_JUDGE_API_BASE` environment variables.')
        DEFAULT_JUDGE_CFG = dict(
            type=OpenAISDK,
            path=os.environ['OC_JUDGE_MODEL'],
            key=os.environ['OC_JUDGE_API_KEY'],
            openai_api_base=[
                os.environ.get('OC_JUDGE_API_BASE',
                               'https://api.openai.com/v1/')
            ],
            meta_template=dict(round=[
                dict(role='HUMAN', api_role='HUMAN'),
                dict(role='BOT', api_role='BOT', generate=True),
            ], ),
            query_per_second=16,
            batch_size=1024,
            temperature=0.001,
            tokenizer_path='gpt-4o-2024-05-13',
            verbose=True,
            max_out_len=16384,
            max_seq_len=49152,
        )

        return DEFAULT_JUDGE_CFG