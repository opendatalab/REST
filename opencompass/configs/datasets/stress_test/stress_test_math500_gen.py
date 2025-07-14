from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import StressMATHDataset, StressTestMATHEvaluator
from opencompass.utils.text_postprocessors import split_boxed_content


math_reader_cfg = dict(input_columns=['question'], output_column='answer')

math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{question}\nAnswer the above questions one by one. Remember to put your final answer within \\boxed{}.'),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

for n_merge in [1, 3, 5, 7, 9]:
    math_eval_cfg = dict(
        evaluator=dict(type=StressTestMATHEvaluator), 
        pred_postprocessor=dict(type=split_boxed_content, n_merge=n_merge)
    )
    globals()[f'math_merge{n_merge}_datasets'] = [
        dict(
            type=StressMATHDataset,
            abbr=f'math500_merge{n_merge}',
            path='math',
            file_name='test_prm800k_500.json',
            n_merge=n_merge,
            reader_cfg=math_reader_cfg,
            infer_cfg=math_infer_cfg,
            eval_cfg=math_eval_cfg,
            mode='singlescore',
        )
    ]
