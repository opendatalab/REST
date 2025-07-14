import copy
from opencompass.utils import get_logger
import copy
import re
from opencompass.datasets.generic import get_final_results, _generic_llmjudge_postprocess

GPQA_TEMPLATE = """
{question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

LCB_TEMPLATE = "{question_content}\n\n{format_prompt}"

def merge_math(dataset, q_key: str = 'question', a_key: str = 'answer', n_merge: int = 5):
    merged_dataset = []
    for i, sample in enumerate(dataset):
        new_sample = copy.deepcopy(sample)
        
        sample_to_merge = []
        for j in range(1, n_merge):
            sample = dataset[(i+j)%len(dataset)]
            sample_to_merge.append(sample)
        sample_to_merge.append(new_sample)

        questions_str = ""
        answers_str = ""
        question_list = []
        answer_list = []
        for j, sample in enumerate(sample_to_merge):
            questions_str += f"Q{j+1}: {sample[q_key]}\n\n"
            answers_str += f"Answer to Q{j+1}: " + "\\boxed{" + f"{sample[a_key]}" + "}\n\n"
            question_list.append(sample[q_key])
            answer_list.append(sample[a_key])

        new_sample[q_key] = questions_str
        new_sample[a_key] = answers_str
        new_sample[f"{q_key}_list"] = question_list
        new_sample[f"{a_key}_list"] = answer_list
        new_sample["q_key"] = q_key
        new_sample["a_key"] = a_key
        merged_dataset.append(new_sample)
    return merged_dataset

def merge_gpqa(dataset, q_key: str = 'question', a_key: str = 'answer', n_merge: int = 5):
    
    merged_dataset = []
    for i, sample in enumerate(dataset):
        new_sample = copy.deepcopy(sample)
        
        sample_to_merge = []
        for j in range(1, n_merge):
            sample = dataset[(i+j)%len(dataset)]
            sample_to_merge.append(sample)
        sample_to_merge.append(new_sample)

        questions_str = ""
        answers_str = ""
        question_list = []
        answer_list = []
        for j, sample in enumerate(sample_to_merge):
            question = GPQA_TEMPLATE.format(question=sample[q_key], A=sample['ABCD'[0]], B=sample['ABCD'[1]], C=sample['ABCD'[2]], D=sample['ABCD'[3]])
            questions_str += f"Q{j+1}: {question}\n\n"
            answers_str += f"Answer to Q{j+1}: " + "\\boxed{" + f"{sample[a_key]}" + "}\n\n"
            question_list.append(sample[q_key])
            answer_list.append(sample[a_key])

        new_sample[q_key] = questions_str
        new_sample[a_key] = answers_str
        new_sample[f"{q_key}_list"] = question_list
        new_sample[f"{a_key}_list"] = answer_list
        new_sample["q_key"] = q_key
        new_sample["a_key"] = a_key
        merged_dataset.append(new_sample)

    return merged_dataset

def merge_livecodebench(dataset, q_key: str = 'question_content', a_key: str = 'question_id', n_merge: int = 2):
    # assert mode in ["infer", "eval"], f"Invalid mode: {mode}"

    merged_dataset = []

    for i, sample in enumerate(dataset):
        this_sample = copy.deepcopy(sample)
        
        sample_to_merge = []
        for j in range(1, n_merge):
            sample = dataset[(i+j)%len(dataset)]
            sample_to_merge.append(sample)
        sample_to_merge.append(this_sample)

        # if mode == "eval":
        #     merged_dataset.extend(copy.deepcopy(sample_to_merge))
        #     continue

        questions_str = "Answer the following questions one by one. Enclose the code for each question within delimiters as follows.\n```python\n# YOUR CODE HERE\n```\n\n\n\n"
        question_list = []
        answer_list = []
        format_prompt_list = []
        for j, sample in enumerate(sample_to_merge):
            answer_list.append(sample[a_key])
            format_prompt_list.append(sample["format_prompt"])
            question = LCB_TEMPLATE.format(question_content=sample[q_key], format_prompt=sample['format_prompt'])
            questions_str += f"Q{j+1}: {question.strip()}\n\n"
            question_list.append(sample[q_key])
        questions_str += "\n\nAnswer the above questions one by one. Enclose the code for each question within delimiters as follows.\n```python\n# YOUR CODE HERE\n```. ### Answer: (use the provided format with backticks)\n\n"
        questions_str = questions_str.strip()
        merged_sample = copy.deepcopy(this_sample)
        merged_sample[q_key] = questions_str
        merged_sample[f"{q_key}_list"] = question_list
        merged_sample[f"{a_key}_list"] = answer_list
        merged_sample["format_prompt"] = format_prompt_list
        merged_sample["q_key"] = q_key
        merged_sample["a_key"] = a_key
        merged_dataset.append(merged_sample)

    return merged_dataset