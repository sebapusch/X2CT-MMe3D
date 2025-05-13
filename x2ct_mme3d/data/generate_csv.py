import pandas as pd

KEYWORDS = [
    # Already in your list and valid
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
    'deformity',

    # Newly added anatomical/pathological findings
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


df = pd.read_csv("indiana_reports.csv")
projections_df = pd.read_csv("indiana_projections.csv")

labels = []
filtered_rows = []

for idx, row in df.iterrows():
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
    elif  any(kw in problem for kw in KEYWORDS) or ('opacity' in problem and 'lung' in problem):
        labels.append(1)
        filtered_rows.append(row)

filtered_df = pd.DataFrame(filtered_rows)
filtered_df['normal'] = labels

print('normal:', len(filtered_df[filtered_df['normal'] == 1]))
print('non normal:', len(filtered_df[filtered_df['normal'] == 0]))

output_path = "output.csv"

filtered_df.to_csv(output_path, index=False)