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
    'deformity'
]


df = pd.read_csv("indiana_reports.csv")

labels = []
filtered_rows = []

for idx, row in df.iterrows():
    problem = str(row['Problems']).lower()

    if 'normal' in problem:
        labels.append(0)
        filtered_rows.append(row)
    elif  any(kw in problem for kw in KEYWORDS) or ('opacity' in problem and 'lung' in problem):
        labels.append(1)
        filtered_rows.append(row)

filtered_df = pd.DataFrame(filtered_rows)
filtered_df['normal'] = labels

output_path = "outptu.csv"

filtered_df.to_csv(output_path, index=False)