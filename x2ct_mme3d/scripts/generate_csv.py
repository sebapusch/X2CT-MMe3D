import os.path

import pandas as pd

KEYWORDS = [
    'atelectasis',
    'pneumonia',
    'pleural effusion',
    'pulmonary edema',
    'pulmonary disease',
    'pulmonary congestion',
    'emphysema',
    'bronchiectasis',
    'pulmonary fibrosis',
    'interstitial lung disease',
    'copd',
    'lung mass',
    'pulmonary mass',
    'hilar enlargement',
    'infiltrate',
    'airspace disease',

    'consolidation',
    'effusion'
    'edema',
    'interstitial',
    'nodular'
    'mass',
    'nodule',
    'honeycombing',
    'fibrotic',
    'reticular',
    'hyperinflation',
    'thickening',

    'sarcoidosis',
    'pneumothorax',
    'tuberculosis',
    'pulmonary hypertension',
    'bronchitis',
    'lung cancer',
    'cavitating lesion',
    'abscess',
    'empyema',
    'ards',
    'silicosis',
    'asbestosis',
]

MISSING_FILES = [
    372,
    812,
    1202,
    1591,
    2000,
    2389,
    2826,
    3218,
    3593,
]

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

RAW_BASE_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_BASE_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

df = pd.read_csv(os.path.join(RAW_BASE_DIR, 'indiana_reports.csv'))
projections_df = pd.read_csv(os.path.join(RAW_BASE_DIR, 'indiana_projections.csv'))

labels = []
filtered_rows = []

for idx, row in df.iterrows():
    if int(row['uid']) in MISSING_FILES:
        print('file known to be missing, skipping')
        continue

    projection = projections_df[(projections_df['uid'] == row['uid']) & (projections_df['projection'] == 'Frontal')]
    if len(projection) == 0:
        print('missing frontal projection, skipping')
        continue

    projection = projections_df[(projections_df['uid'] == row['uid']) & (projections_df['projection'] == 'Lateral')]
    if len(projection) == 0:
        print('missing lateral projection, skipping')
        continue

    problem = str(row['Problems']).lower()

    if 'normal' in problem:
        labels.append(0)
        filtered_rows.append(row)
    elif  (any(kw in problem for kw in KEYWORDS) or
           ('opacity'   in problem and 'lung' in problem) or
           ('deformity' in problem and 'lung' in problem)):
        labels.append(1)
        filtered_rows.append(row)

filtered_df = pd.DataFrame(filtered_rows)
filtered_df['disease'] = labels

disease = len(filtered_df[filtered_df['disease'] == 1])

print('disease:', disease, 'normal:', len(filtered_df) - disease)

output_path = os.path.join(PROCESSED_BASE_DIR, 'indiana_reports.csv')

filtered_df.to_csv(output_path, index=False)