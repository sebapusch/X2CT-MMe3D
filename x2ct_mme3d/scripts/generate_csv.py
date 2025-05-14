from os.path import join, dirname, abspath

import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN = 0.9

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

PROJECT_ROOT = abspath(join(dirname(__file__), '..', '..'))

RAW_BASE_DIR = join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_BASE_DIR = join(PROJECT_ROOT, 'data', 'processed')

df = pd.read_csv(join(RAW_BASE_DIR, 'indiana_reports.csv'))
projections_df = pd.read_csv(join(RAW_BASE_DIR, 'indiana_projections.csv'))

labels = []
filtered_rows = []

skipped = 0

for idx, row in df.iterrows():
    if int(row['uid']) in MISSING_FILES:
        skipped += 1
        continue

    projection = projections_df[(projections_df['uid'] == row['uid']) & (projections_df['projection'] == 'Frontal')]
    if len(projection) == 0:
        skipped += 1
        continue

    projection = projections_df[(projections_df['uid'] == row['uid']) & (projections_df['projection'] == 'Lateral')]
    if len(projection) == 0:
        skipped += 1
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

train, test = train_test_split(
    filtered_df,
    test_size=1 - TRAIN,
    stratify=labels,
    shuffle=True,
)

disease_len = len(filtered_df[filtered_df['disease'] == 1])

print('total:   ', len(filtered_df))
print('----------------------')
print('normal:  ', len(filtered_df) - disease_len)
print('disease: ', disease_len)
print('----------------------')
print('skipped: ', skipped)
print('----------------------')
print('train:   ', len(train))
print('test:    ', len(test))

filtered_df.to_csv(join(PROCESSED_BASE_DIR, 'indiana_reports.csv'), index=False)
train.to_csv(join(PROCESSED_BASE_DIR, 'indiana_reports.train.csv'), index=False)
test.to_csv(join(PROCESSED_BASE_DIR, 'indiana_reports.test.csv'), index=False)