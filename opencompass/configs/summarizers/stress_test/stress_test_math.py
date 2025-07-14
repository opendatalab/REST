# Support for AIME24, AIME25, AMC23, GSM8K and MATH500

summary_groups = sum(
    [v for k, v in locals().items() if k.endswith('_summary_groups')], []
)

summary_groups += [
    {
        'name': 'AMC23_Merge1_Aveage8',
        'subsets':[[f'amc23_merge1_run{idx}', 'accuracy'] for idx in range(8)]
    },
    {
        'name': 'AMC23_Merge3_Aveage8',
        'subsets':[[f'amc23_merge3_run{idx}', 'accuracy'] for idx in range(8)]
    },
    {
        'name': 'AMC23_Merge5_Aveage8',
        'subsets':[[f'amc23_merge5_run{idx}', 'accuracy'] for idx in range(8)]
    },
    {
        'name': 'AMC23_Merge7_Aveage8',
        'subsets':[[f'amc23_merge7_run{idx}', 'accuracy'] for idx in range(8)]
    },
    {
        'name': 'AMC23_Merge9_Aveage8',
        'subsets':[[f'amc23_merge9_run{idx}', 'accuracy'] for idx in range(8)]
    },
    {
        'name': 'AIME2024_Merge1_Aveage8',
        'subsets':[[f'aime2024_merge1_run{idx}', 'accuracy'] for idx in range(8)]
    },
    {
        'name': 'AIME2024_Merge2_Aveage8',
        'subsets':[[f'aime2024_merge2_run{idx}', 'accuracy'] for idx in range(8)]
    },
    {
        'name': 'AIME2024_Merge3_Aveage8',
        'subsets':[[f'aime2024_merge3_run{idx}', 'accuracy'] for idx in range(8)]
    },
    {
        'name': 'AIME2024_Merge4_Aveage8',
        'subsets':[[f'aime2024_merge4_run{idx}', 'accuracy'] for idx in range(8)]
    },
    {
        'name': 'AIME2024_Merge5_Aveage8',
        'subsets':[[f'aime2024_merge5_run{idx}', 'accuracy'] for idx in range(8)]
    },
    
    {
        'name': 'AIME2025_Merge1_Aveage8',
        'subsets':[[f'aime2025_merge1_run{idx}', 'accuracy'] for idx in range(8)]
    },
    {
        'name': 'AIME2025_Merge2_Aveage8',
        'subsets':[[f'aime2025_merge2_run{idx}', 'accuracy'] for idx in range(8)]
    },
    {
        'name': 'AIME2025_Merge3_Aveage8',
        'subsets':[[f'aime2025_merge3_run{idx}', 'accuracy'] for idx in range(8)]
    },
    {
        'name': 'AIME2025_Merge4_Aveage8',
        'subsets':[[f'aime2025_merge4_run{idx}', 'accuracy'] for idx in range(8)]
    },
    {
        'name': 'AIME2025_Merge5_Aveage8',
        'subsets':[[f'aime2025_merge5_run{idx}', 'accuracy'] for idx in range(8)]
    }
]

summarizer = dict(
    dataset_abbrs=[
        ['gsm8k_merge1', 'accuracy'],
        ['gsm8k_merge3', 'accuracy'],
        ['gsm8k_merge6', 'accuracy'],
        ['gsm8k_merge9', 'accuracy'],
        ['gsm8k_merge12', 'accuracy'],
        
        ['math500_merge1', 'accuracy'],
        ['math500_merge3', 'accuracy'],
        ['math500_merge5', 'accuracy'],
        ['math500_merge7', 'accuracy'],
        ['math500_merge9', 'accuracy'],

        ["AMC23_Merge1_Aveage8", "naive_average"],
        ["AMC23_Merge3_Aveage8", "naive_average"],
        ["AMC23_Merge5_Aveage8", "naive_average"],
        ["AMC23_Merge7_Aveage8", "naive_average"],
        ["AMC23_Merge9_Aveage8", "naive_average"],

        ['AIME2024_Merge1_Aveage8', 'naive_average'],
        ['AIME2024_Merge2_Aveage8', 'naive_average'],
        ['AIME2024_Merge3_Aveage8', 'naive_average'],
        ['AIME2024_Merge4_Aveage8', 'naive_average'],
        ['AIME2024_Merge5_Aveage8', 'naive_average'],

        ['AIME2025_Merge1_Aveage8', 'naive_average'],
        ['AIME2025_Merge2_Aveage8', 'naive_average'],
        ['AIME2025_Merge3_Aveage8', 'naive_average'],
        ['AIME2025_Merge4_Aveage8', 'naive_average'],
        ['AIME2025_Merge5_Aveage8', 'naive_average'],
    ],
    summary_groups=summary_groups
)
