from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import StressGPQADataset, StressTestLLMExtractEvaluator
# from opencompass.datasets.stress_test import generic_llmjudge_split_postprocess
from opencompass.models import OpenAISDK
import os
import copy
from opencompass.utils.text_postprocessors import extract_non_reasoning_content


align_prompt = """
Answer the following multiple choice question one by one. Remember to give each answer in the following format: 'ANSWER: \\boxed{{$LETTER}}' (without quotes) where LETTER is one of ABCD.

{question}

Answer the above multiple choice question one by one. Remember to give each answer in the following format: 'ANSWER: \\boxed{{$LETTER}}' (without quotes) where LETTER is one of ABCD.
""".strip()

PROMPT_TEMPLATE="""
Extract the final answers from the given predictions. 

Here are some extraction criteria:
1. Don't try to answer the original question. Your task is to extract the final answer from the prediction as it is, even if it is incorrect.
2. Prediction sometimes involves lengthy thinking processes, you don't need to consider these, just extract the final answer.
3. If there is question that has not been answered, don't answer it yourself. You should set the final answer to None (e.g., \\boxed{{None}}).
4. Ensure the number of final answers you extract is exatly the same as the number of the given questions.
5. Extract the final answer for each question one by one and enclose each final answer within an \\boxed{{}}. For example, if there are five questions, the output should be "Answer to Q1: \\boxed{{answer 1}}\nAnswer to Q2: \\boxed{{answer 2}}\nAnswer to Q3: \\boxed{{answer 3}}\nAnswer to Q4: \\boxed{{answer 4}}\nAnswer to Q5: \\boxed{{answer 5}}".

Here is an example:
<Questions Begin>:
Q1: \"Oh, I know you,\" the ribonucleoprotein particle says to the nascent chain as they meet. \"Pause there for a minute. Let me show you in; you really need some sugar.\"\n\"It seems somewhat rough. I guess this is goodbye; I need to be on my way\", the chain replies. Where did they meet, and where is the chain heading?\n\nA) ribosome to the proteasome\nB) cytosol to the extracellular space\nC) Golgi to the mitochondrion\nD) membrane to the nucleus\n
Q2: S)-4-hydroxycyclohex-2-en-1-one is treated with tert-Butyldimethylsilyl chloride and triethylamine, forming product 1.\n\n1 is treated with Ph2CuLi at low temperature, followed by benzyl bromide, forming product 2.\n\n2 is treated with LDA and iodomethane at low temperature, forming product 3.\n\nProduct 3 is treatd with aqueous HCl, forming final product 4. what is the structure of product 4?\n\nA) (2S,3R,4S,6S)-2-benzyl-4-hydroxy-6-methyl-3-phenylcyclohexan-1-one\nB) (1S,2S,4S)-1-(benzyloxy)-2-methyl-1,2,3,4-tetrahydro-[1,1'-biphenyl]-4-ol\nC) (2S,3S,4S)-2-benzyl-4-hydroxy-2-methyl-3-phenylcyclohexan-1-one\nD) (2R,3R,4S)-2-benzyl-4-hydroxy-2-methyl-3-phenylcyclohexan-1-one\n
Q3: You have an interesting drought-resistant cultivar of barley, which, unfortunately, contains an anti-nutritional compound. This compound makes it impossible to use this cultivar for food purposes. After an extensive investigation, you discover the gene responsible for the synthesis of this compound. This gene consists of five exons and four introns. You decide to use old-school approaches and produce a collection of mutants using EMS chemical mutagen. You sequence a target gene in all mutant lines and discover some changes at the beginning of its sequence in the first exon area. Which of the following mutations will most probably let you eliminate the anti-nutritional compound in the cultivar?\n\nIntact gene:\n5’-ATGTTTCTCGCTGGTACTTCTGTGGATGAACATATTTATTGTCGT…TGA-3’\n\nMutant 1:\n5’-ATGTTCTACGCTGGTACTTCTGTGGATGAACATATTTATTGTCGC…TGA-3’\nMutant 2:\n5’-ATGTTCTAAGCTGGTACTTCTGTGGATGAACATATTTATTGTCGC…TGA-3’\nMutant 3:\n5’-ATGTTTTACGCTGGTGTCACTTCTGTGGATGAACATATTTATTGTCGT…TGA-3’\nMutant 4:\n5’-ATGTTTTACGCTACTTCTGTGGATGAACATATTTATTGTCGT…TGA-3’<b>\n\nA) Mutant 3\nB) Mutant 4\nC) Mutant 1\nD) Mutant 2\n
Q4: Dienes are organic compounds with two adjacent double bonds in their structure, and they exhibit unique reactivity due to their conjugated pi-electron system. They play a significant role in organic chemistry and are involved in various chemical reactions and natural processes.\nAmong the given options which one is the possible reactant (A) for the given reaction also mention the correct sequence of the dienes according to their reactivity ( most reactive to least reactive) B.\nCyclohexene + A ---> 8,8-diiodobicyclo[4.2.0]octan-7-one\n(B) 1. 2,3-dimethylbuta-1,3-diene, 2. (2E,4E)-hexa-2,4-diene, 3. cyclopenta-1,3-diene, 4. (2Z,4Z)-hexa-2,4-diene\n\nA) A = 2,2-diiodoethen-1-one, B = 4, 2, 1, 3\nB) A = 4,4-diiodocyclobut-2-en-1-one, B = 4, 2, 1, 3\nC) A = 2,2-diiodoethen-1-one, B = 3, 1, 2, 4\nD) A = 4,4-diiodocyclobut-2-en-1-one, B = 3, 1, 2, 4\n\n\n\nAnswer the above multiple choice question one by one. Remember to give each answer in the following format: 'ANSWER: \\boxed{{$LETTER}}' (without quotes) where LETTER is one of ABCD.
<Questions End>

<Predictions Begin>: 
### Q1: Ribosomally Directed Reactions\n\nThe question asks where the ribosome meets the nascent chain and where it leaves. Ribosomes are protein-synthetizing molecules that translate mRNA into proteins. During translation, the ribosome binds to the mRNA at the 5' end, where it starts translating the codons. The ribosome moves along the mRNA, peeling off the nascent chain as it goes. The meeting point is the ribosome and the mRNA, and the leaving point is the ribosome and the extracellular space after the mRNA is released. \n\n**ANSWER: B**\n\n---\n\n### Q2: Reaction with Cyclohex-2-en-1-one\n\nThe starting material is S)-4-hydroxycyclohex-2-en-1-one, which is treated with tert-butyl chloride silyl chloride and triethylamine, forming a silyl ether. This is followed by treatment with Ph₂CuLi at low temperature, then benzyl bromide, and finally LDA followed by iodomethane. The final product is a cyclohexanone derivative with a benzyl group and a hydroxyl group.\n\nPh₂CuLi is a strong base that deprotonates the alpha position of the cyclohexenone, forming a conjugate base. This conjugate base reacts with benzyl bromide, adding a benzyl group. LDA, a strong base, further deprotonates the alpha position, and iodomethane adds a methyl group. The silyl ether is removed during the reaction sequence, leaving the final product as a cyclohexanone derivative with a benzyl group and a hydroxyl group.\n\nThe correct reagent (A) is **4,4-diiodoethen-1-one**, and the reactivity order (B) is **4,2,1,3**.\n\n**ANSWER: A**\n\n---\n\n### Q3: Anti-Nutritional Compound Elimination\n\nThe question involves a gene with five exons and four introns, where the anti-nutritional compound is encoded in the first exon. Mutants (A-D) differ at the beginning of the gene. Mutant 1 has no changes, Mutant 2 changes the 5th nucleotide from T to A, Mutant 3 changes it to G, and Mutant 4 changes it to C. The goal is to determine which mutation will eliminate the anti-nutritional compound.\n\nMutant 2 changes the 5th nucleotide from T to A, altering the amino acid at that position. This disrupts the production of the anti-nutritional compound, making it the most likely to eliminate the compound.\n\n**ANSWER: B**\n\n---\n\n### Q4: Dienes Reactions\n\nThe question involves the reactivity of dienes in a reaction sequence. The dienes are:\n\n- **A)** cyclopenta-1,3-diene  \n- **B)** cyclopenta-1,2-diene  \n- **C)** cyclopenta-1,3-diene  \n- **D)** cyclopenta-1,4-diene  \n\nThe reactivity order is determined by the stability of the transition state in conjugate addition reactions. Conjugated dienes (A and C) are more reactive than isolated diene (B). Among the conjugated dienes, **D (cyclopenta-1,4-diene)** is more reactive than **A and C (cyclopenta-1,3-diene)** due to better orbital overlap in the transition state.\n\nThe correct reagent (A) is **4,4-diiodoethen-1-one**, and the reactivity order (B) is **4,2,1,3**.\n\n**ANSWER: B**
<Predictions End>

<Extraction Begin>:
Answer to Q1: \\boxed{{B}}\nAnswer to Q2: \\boxed{{A}}\nAnswer to Q3: \\boxed{{B}}\nAnswer to Q4: \\boxed{{B}}
<Extraction End>

Here is your task. Simply extract the final answers from the given predictions. Don't apologize or correct yourself if there was a mistake in the predictions; we are just trying to extract the final answer. Remember the number of final answers you extract should be exactly the same as the number of the given questions.

<Questions Begin>: \n{question}\n<Questions End>\n\n
<Predictions Begin>: \n{prediction}\n<Predictions End>\n\n

Extract the final answers from the given predictions.
""".strip()

verifier_cfg = dict(
        abbr=os.getenv('VERIFYER_MODEL_NAME', None),
        type=OpenAISDK,
        path=os.getenv('VERIFYER_MODEL_NAME', None), # You need to set your own judge model path
        key=os.getenv('VERIFYER_API_KEY', None), # You need to set your own API key
        openai_api_base=[
            os.getenv('VERIFYER_API_BASE', None), # You need to set your own API base
        ],
        meta_template=dict(
            round=[
                dict(role='HUMAN', api_role='HUMAN'),
                dict(role='BOT', api_role='BOT', generate=True),
            ], 
        ),
        query_per_second=4,
        batch_size=1024,
        temperature=0.001,
        tokenizer_path='gpt-4o-2024-05-13',
        verbose=True,
        max_out_len=128,
        max_seq_len=131072,
)

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
        eval_cfg = dict(
            evaluator=dict(
                type=StressTestLLMExtractEvaluator,
                prompt_template=dict(
                    type=PromptTemplate,
                    template=dict(
                    # begin=[
                    #     dict(
                    #         role='SYSTEM',
                    #         fallback_role='HUMAN',
                    #         prompt="You are a helpful assistant who evaluates the correctness and quality of models' outputs.")
                    # ],
                        round=[
                            dict(
                                role='HUMAN',
                                prompt = PROMPT_TEMPLATE
                            ),
                        ]
                    ),
                ),
                dataset_cfg=dict(
                    type=StressGPQADataset,
                    path='gpqa',
                    n_merge=n_merge,
                    name=gpqa_subsets[split],
                    reader_cfg=gpqa_reader_cfg,
                ),
                judge_cfg=verifier_cfg,
                # dict_postprocessor=dict(type=generic_llmjudge_split_postprocess, n_merge=n_merge),
            ),
            pred_role='BOT',
            pred_postprocessor=dict(type=extract_non_reasoning_content)
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
                eval_cfg=copy.deepcopy(eval_cfg),
                mode='singlescore',
            )
        )
