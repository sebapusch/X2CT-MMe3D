import pandas as pd

df = pd.read_csv('indiana_reports.csv')

output = []
normal = []

for i, problem in enumerate(df['Problems']):
    problem = problem.lower()


    if 'normal' in problem or 'atelectasis' in problem or 'opacity' in problem or 'pleural effusion' in problem or 'pulmonary edema' in problem or 'pulmonary congestion' in problem or 'pulmonary emphysema' in problem or 'pneumonia' in problem or 'lung' in problem:
        normal.append(1 if 'normal' in problem else 0)
        output.append(df.iloc[i])

out = pd.DataFrame(output)
out['normal'] = normal


out.to_csv('output.csv', index=False)

