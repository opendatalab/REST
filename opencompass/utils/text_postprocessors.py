import re
from typing import Callable, Optional, Union, List

from opencompass.registry import TEXT_POSTPROCESSORS


@TEXT_POSTPROCESSORS.register_module('general')
def general_postprocess(text: str) -> str:
    # Cut off the first newline, period, or comma
    truncated_text = re.split(r'[\n.,]', text, 1)[0]

    # Remove punctuation
    no_punctuation = re.sub(r'[^\w\s]', '', truncated_text)

    # Remove article
    no_articles = re.sub(r'\b(a|an|the)\b',
                         '',
                         no_punctuation,
                         flags=re.IGNORECASE)

    # Remove duplicated blank spaces
    cleaned_text = re.sub(r'\s+', ' ', no_articles).strip()

    return cleaned_text


@TEXT_POSTPROCESSORS.register_module('general_cn')
def general_cn_postprocess(text: str) -> str:
    truncated_text = re.split(r'[\n.,]', text, 1)[0]

    no_punctuation = re.sub(r'[^\w\s]', '', truncated_text)

    no_articles = re.sub(r'\b(a|an|the)\b',
                         '',
                         no_punctuation,
                         flags=re.IGNORECASE)

    cleaned_text = re.sub(r'\s+', ' ', no_articles).strip()
    import jieba

    cleaned_text = ' '.join(jieba.cut(text))
    return cleaned_text


@TEXT_POSTPROCESSORS.register_module('first-capital')
def first_capital_postprocess(text: str) -> str:
    for t in text:
        if t.isupper():
            return t
    return ''


@TEXT_POSTPROCESSORS.register_module('last-capital')
def last_capital_postprocess(text: str) -> str:
    for t in text[::-1]:
        if t.isupper():
            return t
    return ''


@TEXT_POSTPROCESSORS.register_module('think_pred')
def think_pred_postprocess(
    prediction: str,
    re_pattern: str,
) -> str:
    match = re.search(re_pattern, prediction)
    if match:
        return match.group(1).strip()
    else:
        return prediction


def first_option_postprocess(text: str, options: str, cushion=True) -> str:
    """Find first valid option for text."""

    # yapf: disable
    # flake8: noqa: W605
    patterns = [
        f'答案是?\s*([{options}])',
        f'答案是?\s*：\s*([{options}])',
        f'答案是?\s*:\s*([{options}])',
        f'答案选项应?该?是\s*([{options}])',
        f'答案选项应?该?为\s*([{options}])',
        f'答案应该?是\s*([{options}])',
        f'答案应该?选\s*([{options}])',
        f'答案选项为?\s*：\s*([{options}])',
        f'答案选项为?\s+\(?\*?\*?([{options}])\*?\*?\)?',
        f'答案选项是?\s*:\s*([{options}])',
        f'答案为\s*([{options}])',
        f'答案选\s*([{options}])',
        f'选择?\s*([{options}])',
        f'故选?\s*([{options}])'
        f'只有选?项?\s?([{options}])\s?是?对',
        f'只有选?项?\s?([{options}])\s?是?错',
        f'只有选?项?\s?([{options}])\s?不?正确',
        f'只有选?项?\s?([{options}])\s?错误',
        f'说法不?对选?项?的?是\s?([{options}])',
        f'说法不?正确选?项?的?是\s?([{options}])',
        f'说法错误选?项?的?是\s?([{options}])',
        f'([{options}])\s?是正确的',
        f'([{options}])\s?是正确答案',
        f'选项\s?([{options}])\s?正确',
        f'所以答\s?([{options}])',
        f'所以\s?([{options}][.。$]?$)',
        f'所有\s?([{options}][.。$]?$)',
        f'[\s，：:,]([{options}])[。，,\.]?$',
        f'[\s，,：:][故即]([{options}])[。\.]?$',
        f'[\s，,：:]因此([{options}])[。\.]?$',
        f'[是为。]\s?([{options}])[。\.]?$',
        f'因此\s?([{options}])[。\.]?$',
        f'显然\s?([{options}])[。\.]?$',
        f'答案是\s?(\S+)(?:。|$)',
        f'答案应该是\s?(\S+)(?:。|$)',
        f'答案为\s?(\S+)(?:。|$)',
        f'(?i)ANSWER\s*:\s*([{options}])',
        f'[Tt]he answer is:?\s+\(?([{options}])\)?',
        f'[Tt]he answer is:?\s+\(?\*?\*?([{options}])\*?\*?\)?',
        f'[Tt]he answer is option:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is option:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is:?.*?boxed{{([{options}])}}',
        f'[Tt]he correct option is:?.*?boxed{{([{options}])}}',
        f'[Tt]he correct answer option is:?.*?boxed{{([{options}])}}',
        f'[Tt]he answer to the question is:?\s+\(?([{options}])\)?',
        f'^选项\s?([{options}])',
        f'^([{options}])\s?选?项',
        f'(\s|^)[{options}][\s。，,：:\.$]',
        f'1.\s?(.*?)$',
        f'1.\s?([{options}])[.。$]?$',
    ]
    cushion_patterns = [
        f'([{options}]):',
        f'([{options}])',
    ]
    # flake8: noqa
    # yapf: enable

    if cushion:
        patterns.extend(cushion_patterns)
    for pattern in patterns:
        text = text.strip()
        match = re.search(pattern, text, re.DOTALL)
        if match:
            if match.group(1) is not None and match.group(1) != '':
                outputs = match.group(1)
            else:
                outputs = match.group(0)
            for i in options:
                if i in outputs:
                    return i
    return ''


@TEXT_POSTPROCESSORS.register_module('first-capital-multi')
def first_capital_postprocess_multi(text: str) -> str:
    match = re.search(r'([A-D]+)', text)
    if match:
        return match.group(1)
    return ''


def last_option_postprocess(text: str, options: str) -> str:
    match = re.findall(rf'([{options}])', text)
    if match:
        return match[-1]
    return ''


def first_number_postprocess(text: str) -> float:
    """Return the first number in a string."""
    # regex pattern to match numbers (both integers and decimals)
    pattern = r'(-?\d*\.?\d+)'

    # search the string for the pattern
    match = re.search(pattern, text)

    # if a match is found, return it. Otherwise, return None.
    return float(match.group(1)) if match else None


@TEXT_POSTPROCESSORS.register_module('multiple-select')
def multiple_select_postprocess(text: str) -> str:
    ret = set([t for t in text if t.isupper()])
    return ''.join(sorted(ret))


@TEXT_POSTPROCESSORS.register_module('specific-xml-tag')
def xml_tag_postprocessor(text, tag):
    """Extracts content enclosed within a specified XML-style tag from a
    string.

    Args:
        texts: The input string containing XML-style tags.
        tag: The XML-style tag to extract content from (e.g., "<conclude>").  Must include the angle brackets.

    Returns:
        The content enclosed within the specified tag, or None if the tag is not found.
    """

    # Use a regular expression to find the content within the specified tag.  This handles cases where the tag might appear multiple times.
    matches = re.findall(
        rf'{tag}(.*?)</{tag[1:-1]}>', text,
        re.DOTALL)  # re.DOTALL allows . to match newline characters

    if matches:
        # Only keep the last one
        output = matches[-1].strip(
        )  # Extract the content and remove leading/trailing whitespace
    else:
        output = 'NO ANSWER FOUND'

    return output


def general_eval_wrapper_postprocess(text: str,
                                     postprocess: Optional[Union[
                                         str, Callable]] = None,
                                     **kwargs) -> str:
    """Wrapper for eval text repr. Especially for chatglmpro.

    Args:
        text(str): Text to be postprocessed.
        postprocess(Callable, optional): Original post processing function.
            Defaults to None.
        **kwargs: Other necessary kwargs for post processing function.
    """
    try:
        text = eval(text)
    except Exception:
        # in case empty input or other error, skip eval
        pass

    if postprocess:
        if isinstance(postprocess, str):
            postprocess = TEXT_POSTPROCESSORS.get(postprocess)
        return postprocess(text, **kwargs)
    else:
        return text


def match_answer_pattern(response_text: str, answer_pattern: str):
    match = re.search(answer_pattern, response_text)
    extracted_answer = match.group(1) if match else ''
    return extracted_answer


@TEXT_POSTPROCESSORS.register_module('extract-non-reasoning-content')
def extract_non_reasoning_content(
    text: str,
    think_start_token: str = '<think>',
    think_end_token: str = '</think>',
) -> str:
    """Extract content after the last reasoning tag from text.

    When only end token is present, returns content after the end token.
    When both tokens are present, removes all content between start and end tokens.

    Args:
        text (str): Input text containing reasoning tags.
        think_start_token (str, optional): Start token for reasoning section. Defaults to '<think>'.
        think_end_token (str, optional): End token for reasoning section. Defaults to '</think>'.

    Returns:
        str: Processed text after removing reasoning sections.

    Examples:
        >>> # When only end token exists
        >>> text = "This is a test.</think> How are you?"
        >>> extract_non_reasoning_content(text)
        'How are you?'

        >>> # When both tokens exist
        >>> text = "Start<think>reasoning here</think> End"
        >>> extract_non_reasoning_content(text)
        'Start End'
    """
    # If text contains only end token, split by end token and take the last part
    if think_start_token not in text and think_end_token in text:
        return text.split(think_end_token)[-1].strip()

    # Original behavior for complete tag pairs
    reasoning_regex = re.compile(rf'{think_start_token}(.*?){think_end_token}',
                                 re.DOTALL)
    non_reasoning_content = reasoning_regex.sub('', text).strip()
    return non_reasoning_content

def extract_boxed(string, pattern='\\boxed'):
    idx = string.find(pattern)
    if idx < 0:
        idx = string.find('\\fbox')
        if idx < 0:
            return None, None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == '{':
            num_left_braces_open += 1
        if string[i] == '}':
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
        right_string = None
    else:
        retval = string[idx:right_brace_idx + 1]
        if right_brace_idx + 1 >= len(string):
            right_string = None
        else:
            right_string = string[right_brace_idx + 1:]
    return retval, right_string

@TEXT_POSTPROCESSORS.register_module('split-boxed-content')
def split_boxed_content(
    text: List[str],
    pattern: str = '\\boxed',
    n_merge: int = 1,
) -> str:
    """Split text by the given pattern.

    Args:
        text (str): Input text containing reasoning tags.
        pattern (str, optional): Pattern to split text. Defaults to r'\boxed'.
        n_merge (int, optional): Number of answers to merge. Defaults to 1.
    Returns:
        str: Processed text after removing reasoning sections.

    Examples:
        >>> # When only end token exists
        >>> text = "Answer to Q1: \\boxed{1} Answer to Q2: \\boxed{2} Answer to Q3: \\boxed{3}"
        >>> split_answer_content(text)
        ['Answer to Q1: \\boxed{1}', 'Answer to Q2: \\boxed{2}', 'Answer to Q3: \\boxed{3}']
    """
    text = extract_non_reasoning_content(text)

    remain_text = text
    answer_list = []
    q_idx = 0
    while remain_text is not None:
        if pattern not in remain_text:
            break
        q_idx += 1
        answer, remain_text = extract_boxed(remain_text, pattern)
        if not answer:
            answer = "\\boxed{None}"
        answer_list.append(answer)
    if len(answer_list) > n_merge:
        answer_list = answer_list[:n_merge]
    elif len(answer_list) < n_merge:
        for q_idx in range(len(answer_list) + 1, n_merge + 1):
            # answer_list.append("Answer to Q" + str(q_idx) + ": \\boxed{None}")
            answer_list.append("\\boxed{None}")
    return answer_list

@TEXT_POSTPROCESSORS.register_module('split-code-content')
def split_code_content(
    text: List[str],
    pattern: str = r'```python(.*?)```',
    n_merge: int = 1,
) -> str:
    """Split text by the given pattern.

    Args:
        text (str): Input text containing reasoning tags.
        pattern (str, optional): Pattern to split text. Defaults to r'\boxed'.
        n_merge (int, optional): Number of answers to merge. Defaults to 1.
    Returns:
        str: Processed text after removing reasoning sections.

    Examples:
        >>> # When only end token exists
        >>> text = "Answer to Q1: \\boxed{1} Answer to Q2: \\boxed{2} Answer to Q3: \\boxed{3}"
        >>> split_answer_content(text)
        ['Answer to Q1: \\boxed{1}', 'Answer to Q2: \\boxed{2}', 'Answer to Q3: \\boxed{3}']
    """
    text = extract_non_reasoning_content(text)

    matches = re.findall(pattern, text, re.DOTALL)
    answer_list = [f"```python\n{match.strip()}\n```" for match in matches]
    if len(answer_list) > n_merge:
        answer_list = answer_list[:n_merge]
    elif len(answer_list) < n_merge:
        for _ in range(len(answer_list) + 1, n_merge + 1):
            answer_list.append(f"```python ```")
    assert len(answer_list) == n_merge, f"len(answer_list) == {len(answer_list)}, n_merge == {n_merge}"
    return answer_list


@TEXT_POSTPROCESSORS.register_module('extract-non-reasoning-content-and-apply-template')
def extract_non_reasoning_content_and_apply_template(
    text: str,
    question: str,
    template: str,
    think_start_token: str = '<think>',
    think_end_token: str = '</think>',
) -> str:
    # If text contains only end token, split by end token and take the last part
    if think_start_token not in text and think_end_token in text:
        return text.split(think_end_token)[-1].strip()

    # Original behavior for complete tag pairs
    reasoning_regex = re.compile(rf'{think_start_token}(.*?){think_end_token}',
                                 re.DOTALL)
    non_reasoning_content = reasoning_regex.sub('', text).strip()
    return template.format(question=question, prediction=non_reasoning_content)