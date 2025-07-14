import os
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import StressAime2024Dataset, StressTestMATHEvaluator
from opencompass.utils.text_postprocessors import split_boxed_content

aime2024_reader_cfg = dict(
    input_columns=['question'], 
    output_column='answer'
)

aime2024_infer_cfg = dict(
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
    aime2024_infer_cfg["prompt_template"]["template"]["round"].insert(0, dict(role='SYSTEM', prompt='detailed thinking on'))

for n_merge in range(1, 6):
    aime2024_eval_cfg = dict(
        evaluator=dict(type=StressTestMATHEvaluator), pred_postprocessor=dict(type=split_boxed_content, n_merge=n_merge)
    )
    globals()[f'aime2024_merge{n_merge}_datasets'] = [
        dict(
            abbr=f'aime2024_merge{n_merge}_run{idx}',
            type=StressAime2024Dataset,
            path=f'aime2024/test.jsonl',
            n_merge=n_merge,
            reader_cfg=aime2024_reader_cfg,
            infer_cfg=aime2024_infer_cfg,
            eval_cfg=aime2024_eval_cfg,
            mode='singlescore',
        )
        for idx in range(8)
    ]
