# Support 32B LRMs

from opencompass.models import (
    TurboMindModelwithChatTemplate,
)

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

models += [
    # You can comment out the models you don't want to evaluate
    # All models use sampling mode
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='deepseek-ai--DeepSeek-R1-Distill-Qwen-32B-turbomind',
        path='deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
        engine_config=dict(session_len=34768, max_batch_size=128, tp=4),
        gen_config=dict(
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.95,
                        max_new_tokens=32768),
        max_seq_len=34768,
        max_out_len=32768,
        batch_size=128,
        run_cfg=dict(num_gpus=4),
    ),
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='Qwen--QwQ-32B-turbomind',
        path='Qwen/QwQ-32B',
        engine_config=dict(session_len=34768, max_batch_size=128, tp=4),
        gen_config=dict(
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.95,
                        max_new_tokens=32768),
        max_seq_len=34768,
        max_out_len=32768,
        batch_size=128,
        run_cfg=dict(num_gpus=4),
    ),
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='inclusionAI--AReaL-boba-SFT-32B-turbomind',
        path='inclusionAI/AReaL-boba-SFT-32B',
        engine_config=dict(session_len=34768, max_batch_size=128, tp=4),
        gen_config=dict(
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.95,
                        max_new_tokens=32768),
        max_seq_len=34768,
        max_out_len=32768,
        batch_size=128,
        run_cfg=dict(num_gpus=4),
    ),
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='qihoo360--Light-R1-32B-DS-turbomind',
        path='qihoo360/Light-R1-32B-DS',
        engine_config=dict(session_len=34768, max_batch_size=128, tp=4),
        gen_config=dict(
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.95,
                        max_new_tokens=32768),
        max_seq_len=34768,
        max_out_len=32768,
        batch_size=128,
        run_cfg=dict(num_gpus=4),
    ),
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='simplescaling--s1.1-32B-turbomind',
        path='simplescaling/s1.1-32B',
        engine_config=dict(session_len=34768, max_batch_size=128, tp=4),
        gen_config=dict(
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.95,
                        max_new_tokens=32768),
        max_seq_len=34768,
        max_out_len=32768,
        batch_size=128,
        run_cfg=dict(num_gpus=4),
    ),
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='open-thoughts--OpenThinker2-32B-turbomind',
        path='open-thoughts/OpenThinker2-32B',
        engine_config=dict(session_len=34768, max_batch_size=128, tp=4),
        gen_config=dict(
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.95,
                        max_new_tokens=32768),
        max_seq_len=34768,
        max_out_len=32768,
        batch_size=128,
        run_cfg=dict(num_gpus=4),
    ),
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='Open-Reasoner-Zero--Open-Reasoner-Zero-32B-turbomind',
        path='Open-Reasoner-Zero/Open-Reasoner-Zero-32B',
        engine_config=dict(session_len=34768, max_batch_size=128, tp=4),
        gen_config=dict(
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.95,
                        max_new_tokens=32768),
        max_seq_len=34768,
        max_out_len=32768,
        batch_size=128,
        run_cfg=dict(num_gpus=4),
    ),


    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='hkust-nlp--Qwen-2.5-32B-SimpleRL-Zoo-turbomind',
        path='hkust-nlp/Qwen-2.5-32B-SimpleRL-Zoo',
        engine_config=dict(session_len=34768, max_batch_size=128, tp=4),
        gen_config=dict(
                        do_sample=False,
                        temperature=0.0,
                        top_p=1.0,
                        max_new_tokens=32768),
        max_seq_len=34768,
        max_out_len=32768,
        batch_size=128,
        run_cfg=dict(num_gpus=4),
    ),
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='Qwen--Qwen2.5-32B-Instruct-turbomind',
        path='Qwen/Qwen2.5-32B-Instruct',
        engine_config=dict(session_len=34768, max_batch_size=128, tp=4),
        gen_config=dict(
                        do_sample=False,
                        temperature=0.0,
                        top_p=1.0,
                        max_new_tokens=32768),
        max_seq_len=34768,
        max_out_len=32768,
        batch_size=128,
        run_cfg=dict(num_gpus=4),
    ),
]
