import argparse
from os.path import join

import torchio
from tqdm import tqdm

from perx2ct.save_to_volume import save
from x2ct_mme3d.data.dataset import CtDataset

def main(args: argparse.Namespace):
    transform = torchio.Compose([
        torchio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5)),
        torchio.ZNormalization(),
        torchio.Resample((1.0, 1.0, 1.0)),
        torchio.CropOrPad((64, 128, 128)),
    ])

    dataset = CtDataset(args.ct_dir,
                        reports_csv_path=args.csv_reports_path,
                        projections_csv_path=args.csv_projections_path)

    for i in tqdm(range(len(dataset))):
        out = transform(dataset[i][0]['ct'])

        report = dataset.reports.iloc[i]
        projection = dataset.projections[(dataset.projections['uid'] == report['uid']) &
                                      (dataset.projections['projection'] == 'Volume')].iloc[0]

        save(out.cpu().numpy(), join(args.save_dir, str(projection['filename'])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess CT')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--ct_dir', type=str, required=True)
    parser.add_argument('--csv_reports_path', type=str, required=True)
    parser.add_argument('--csv_projections_path', type=str, required=True)
    main(parser.parse_args())
