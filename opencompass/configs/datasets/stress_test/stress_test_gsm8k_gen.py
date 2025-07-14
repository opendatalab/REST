from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import StressGSM8KDataset, StressTestMATHEvaluator
from opencompass.utils.text_postprocessors import split_boxed_content

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{question}\nAnswer the above questions one by one. Remember to put your final answer within \\boxed{}.'),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

for n_merge in [1, 3, 6, 9, 12]:
    gsm8k_eval_cfg = dict(
        evaluator=dict(type=StressTestMATHEvaluator), 
        pred_postprocessor=dict(type=split_boxed_content, n_merge=n_merge)
    )
    globals()[f'gsm8k_merge{n_merge}_datasets'] = [
        dict(
            abbr=f'gsm8k_merge{n_merge}',
            type=StressGSM8KDataset,
            path=f'gsm8k',
            n_merge=n_merge,
            reader_cfg=gsm8k_reader_cfg,
            infer_cfg=gsm8k_infer_cfg,
            eval_cfg=gsm8k_eval_cfg,
        )
    ]
