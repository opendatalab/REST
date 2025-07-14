import os
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import StressAMC23Dataset, StressTestMATHEvaluator
from opencompass.utils.text_postprocessors import split_boxed_content

amc23_reader_cfg = dict(
    input_columns=['question'], 
    output_column='answer'
)

amc23_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{question}\nAnswer the above questions one by one. Remember to put your final answer within \\boxed{}.'),
            ],
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)
if os.environ.get('DETAILED_THINK_ON', '0') == '1':
    amc23_infer_cfg["prompt_template"]["template"]["round"].insert(0, dict(role='SYSTEM', prompt='detailed thinking on'))


for n_merge in [1,3,5,7,9]:
    amc23_eval_cfg = dict(
        evaluator=dict(type=StressTestMATHEvaluator), pred_postprocessor=dict(type=split_boxed_content, n_merge=n_merge)
    )
    globals()[f'amc23_merge{n_merge}_datasets'] = [
        dict(
            abbr=f'amc23_merge{n_merge}_run{idx}',
            type=StressAMC23Dataset,
            path='amc23/test.jsonl',
            n_merge=n_merge,
            reader_cfg=amc23_reader_cfg,
            infer_cfg=amc23_infer_cfg,
            eval_cfg=amc23_eval_cfg,
            mode='singlescore',
        )
        for idx in range(8)
    ]