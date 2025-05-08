import os
import numpy as np
import nibabel as nib
import h5py

def save_nifti(volume, out_path: str):
    affine = np.eye(4)
    nii = nib.Nifti1Image(volume.astype(np.float32), affine)
    nib.save(nii, out_path)

def save_npy(volume, out_path: str):
    np.save(out_path, volume.astype(np.float32))

def save_h5(volume, out_path: str, dataset_name='ct'):
    with h5py.File(out_path, 'w') as f:
        f.create_dataset(dataset_name, data=volume.astype(np.float32))

def save(volume, out_path: str):
    ext = os.path.splitext(out_path)[-1].lower()

    if ext == '.nii' or ext == '.nii.gz':
        save_nifti(volume, out_path)
    elif ext == '.npy':
        save_npy(volume, out_path)
    elif ext == '.h5' or ext == '.hdf5':
        save_h5(volume, out_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
