# Support 1.5B LRMs

from opencompass.models import (
    TurboMindModelwithChatTemplate,
)

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

models += [
    # You can comment out the models you don't want to evaluate
    # All models use sampling mode
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B-turbomind',
        path='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
        engine_config=dict(session_len=34768, max_batch_size=128, tp=1),
        gen_config=dict(
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.95,
                        max_new_tokens=32768),
        max_seq_len=34768,
        max_out_len=32768,
        batch_size=128,
        run_cfg=dict(num_gpus=1),
    ),
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='agentica-org--DeepScaleR-1.5B-Preview-turbomind',
        path='agentica-org/DeepScaleR-1.5B-Preview',
        engine_config=dict(session_len=34768, max_batch_size=128, tp=1),
        gen_config=dict(
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.95,
                        max_new_tokens=32768),
        max_seq_len=34768,
        max_out_len=32768,
        batch_size=128,
        run_cfg=dict(num_gpus=1),
    ),
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='l3lab--L1-Qwen-1.5B-Exact-turbomind',
        path='l3lab/L1-Qwen-1.5B-Exact',
        engine_config=dict(session_len=34768, max_batch_size=128, tp=1),
        gen_config=dict(
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.95,
                        max_new_tokens=32768),
        max_seq_len=34768,
        max_out_len=32768,
        batch_size=128,
        run_cfg=dict(num_gpus=1),
    ),
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='l3lab--L1-Qwen-1.5B-Max-turbomind',
        path='l3lab/L1-Qwen-1.5B-Max',
        engine_config=dict(session_len=34768, max_batch_size=128, tp=1),
        gen_config=dict(
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.95,
                        max_new_tokens=32768),
        max_seq_len=34768,
        max_out_len=32768,
        batch_size=128,
        run_cfg=dict(num_gpus=1),
    ),


    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='Qwen--Qwen2.5-Math-1.5B-Instruct-turbomind',
        path='Qwen/Qwen2.5-Math-1.5B-Instruct',
        engine_config=dict(session_len=34768, max_batch_size=128, tp=1),
        gen_config=dict(
                        do_sample=False,
                        temperature=0.0,
                        top_p=1.0,
                        max_new_tokens=32768),
        max_seq_len=34768,
        max_out_len=32768,
        batch_size=128,
        run_cfg=dict(num_gpus=1),
    ),
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='Qwen--Qwen2.5-1.5B-Instruct-turbomind',
        path='Qwen/Qwen2.5-1.5B-Instruct',
        engine_config=dict(session_len=34768, max_batch_size=128, tp=1),
        gen_config=dict(
                        do_sample=False,
                        temperature=0.0,
                        top_p=1.0,
                        max_new_tokens=32768),
        max_seq_len=34768,
        max_out_len=32768,
        batch_size=128,
        run_cfg=dict(num_gpus=1),
    ),
]
