summary_groups = sum(
    [v for k, v in locals().items() if k.endswith('_summary_groups')], []
)

summarizer = dict(
    dataset_abbrs=[
        ['GPQA_diamond', 'accuracy'],
        ['GPQA_diamond_merge1', 'accuracy'],
        ['GPQA_diamond_merge2', 'accuracy'],
        ['GPQA_diamond_merge3', 'accuracy'],
        ['GPQA_diamond_merge4', 'accuracy'],
        ['GPQA_diamond_merge5', 'accuracy'],
        # ['GPQA_diamond_merge6', 'accuracy'],
        # ['GPQA_diamond_merge7', 'accuracy'],
        # ['GPQA_diamond_merge8', 'accuracy'],
        # ['GPQA_diamond_merge9', 'accuracy'],
        # ['GPQA_diamond_merge10', 'accuracy'],
    ],
    summary_groups=summary_groups
)