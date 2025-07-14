from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import (
    StressLCBCodeGenerationDataset,
    StressTestLCBCodeGenerationEvaluator,
)
from opencompass.utils.text_postprocessors import split_code_content

lcb_code_generation_reader_cfg = dict(
    input_columns=[
        'question_content',
        'format_prompt',
    ],
    # output_column='evaluation_sample',
    output_column='question_id',
)

# Code Generation Tasks
lcb_code_generation_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{question_content}'
                )
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

for n_merge in range(1, 6):
    lcb_code_generation_eval_cfg = dict(
        evaluator=dict(
            type=StressTestLCBCodeGenerationEvaluator,
            num_process_evaluate=32,
            timeout=6,
            release_version='release_v5',
            start_date='2024-08-01',
            end_date='2025-02-01'
        ),
        pred_role='BOT',
        pred_postprocessor=dict(type=split_code_content, n_merge=n_merge)
    )
    globals()[f"LCBCodeGeneration_merge{n_merge}_dataset"] = dict(
        type=StressLCBCodeGenerationDataset,
        abbr=f'lcb_code_generation_merge{n_merge}',
        path='opencompass/code_generation_lite',
        reader_cfg=lcb_code_generation_reader_cfg,
        infer_cfg=lcb_code_generation_infer_cfg,
        eval_cfg=lcb_code_generation_eval_cfg,
        n_merge=n_merge,
        local_mode=False,
        release_version='release_v5',
        start_date='2024-08-01',
        end_date='2025-02-01'
    )

    globals()[f"LCB_merge{n_merge}_datasets"] = [
        globals()[f"LCBCodeGeneration_merge{n_merge}_dataset"],
    ]
