# Support Any HF Models
import os
from opencompass.models import (
    TurboMindModelwithChatTemplate,
)

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

MODEL_NAME = os.environ.get("MODEL_NAME", "nvidia/Llama-3.1-Nemotron-Nano-8B-v1")
TEMPERATURE = os.environ.get("TEMPERATURE", 0.6)
TOP_P = os.environ.get("TOP_P", 0.95)
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", 34768))
MAX_OUT_LEN = int(os.environ.get("MAX_OUT_LEN", 32768))
TP_SIZE = int(os.environ.get("TP_SIZE", 1))

models += [
    # You can comment out the models you don't want to evaluate
    # All models use sampling mode
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr=f'{MODEL_NAME.replace("/", "--")}-turbomind',
        path=MODEL_NAME,
        engine_config=dict(session_len=MAX_SEQ_LEN, max_batch_size=128, tp=TP_SIZE),
        gen_config=dict(
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_new_tokens=MAX_OUT_LEN),
        max_seq_len=MAX_SEQ_LEN,
        max_out_len=MAX_OUT_LEN,
        batch_size=128,
        run_cfg=dict(num_gpus=TP_SIZE),
    ),
]
