from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import StressGPQADataset, StressTestMATHEvaluator
from opencompass.utils.text_postprocessors import split_boxed_content
# openai_simple_eval prompt
align_prompt = """
Answer the following multiple choice question one by one. Remember to give each answer in the following format: 'ANSWER: \\boxed{{$LETTER}}' (without quotes) where LETTER is one of ABCD.

{question}

Answer the above multiple choice question one by one. Remember to give each answer in the following format: 'ANSWER: \\boxed{{$LETTER}}' (without quotes) where LETTER is one of ABCD.
""".strip()

gpqa_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer')

gpqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt=align_prompt),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

gpqa_datasets = []
gpqa_subsets = {
    # 'extended': 'gpqa_extended.csv',
    # 'main': 'gpqa_main.csv',
    'diamond': 'gpqa_diamond.csv'
}

for n_merge in range(1, 6):
    globals()[f'gpqa_merge{n_merge}_datasets'] = []
    for split in list(gpqa_subsets.keys()):
        gqpa_eval_cfg = dict(
            evaluator=dict(type=StressTestMATHEvaluator), 
            pred_postprocessor=dict(type=split_boxed_content, n_merge=n_merge)
        )
        globals()[f'gpqa_merge{n_merge}_datasets'].append(
            dict(
                abbr='GPQA_' + split + f'_merge{n_merge}',
                type=StressGPQADataset,
                path='gpqa',
                n_merge=n_merge,
                name=gpqa_subsets[split],
                reader_cfg=gpqa_reader_cfg,
                infer_cfg=gpqa_infer_cfg,
                eval_cfg=gqpa_eval_cfg,
                mode='singlescore',
            )
        )
