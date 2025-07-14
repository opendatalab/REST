from opencompass.models import OpenAISDK
import os
models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

MODEL_NAME = os.environ.get("MODEL_NAME", "deepseek-reasoner")
MAX_SEQ_LEN = os.environ.get("MAX_SEQ_LEN", 34768)
MAX_OUT_LEN = os.environ.get("MAX_OUT_LEN", 32768)
API_KEY = os.environ.get("OPENAI_API_KEY", None)
API_BASE = os.environ.get("OPENAI_API_BASE", None)
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.6))

models += [
    # You can comment out the models you don't want to evaluate
    # All models use sampling mode
    dict(
        type=OpenAISDK,
        abbr=MODEL_NAME,
        path=MODEL_NAME,
        key=API_KEY,
        openai_api_base=API_BASE,
        meta_template=dict(
            round=[
                dict(role='HUMAN', api_role='HUMAN'),
                dict(role='BOT', api_role='BOT', generate=True),
            ], 
        ),
        query_per_second=1,
        batch_size=1,
        temperature=TEMPERATURE,
        max_seq_len=MAX_SEQ_LEN,
        max_out_len=MAX_OUT_LEN,
        verbose=True,
        retry=10,
    ),
]
