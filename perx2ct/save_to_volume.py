import os
import argparse
import numpy as np
import imageio
try:
    import nibabel as nib
    HAVE_NIBABEL = True
except ImportError:
    HAVE_NIBABEL = False
try:
    import SimpleITK as sitk
    HAVE_SITK = True
except ImportError:
    HAVE_SITK = False

def load_slices(input_dir, prefix, start, end):
    slices = []
    for i in range(start, end + 1):
        fname = os.path.join(input_dir, f"{prefix}{i:03d}.png")
        img = imageio.imread(fname)
        if img.ndim == 3:
            # assume RGB -> convert to grayscale by first channel
            img = img[..., 0]
        slices.append(img)
    return np.stack(slices, axis=0)  # shape (Z, H, W)

def save_nifti(volume, out_path):
    if not HAVE_NIBABEL:
        raise ImportError("nibabel is not installed. Please pip install nibabel.")
    affine = np.eye(4)
    nii = nib.Nifti1Image(volume.astype(np.float32), affine)
    nib.save(nii, out_path)
    print(f"Saved NIfTI to {out_path}")

def save_mhd(volume, out_path):
    if not HAVE_SITK:
        raise ImportError("SimpleITK is not installed. Please pip install SimpleITK.")
    img = sitk.GetImageFromArray(volume.astype(np.float32))
    sitk.WriteImage(img, out_path)
    print(f"Saved MHD to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Stack PNG slices into a CT volume file")
    parser.add_argument("--input_dir", required=True, help="Directory containing slice PNGs")
    parser.add_argument("--prefix", default="axial_", help="Filename prefix before the slice index")
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, default=127, help="End index (inclusive)")
    parser.add_argument("--format", choices=["nii", "mhd"], default="nii", help="Output volume format")
    parser.add_argument("--out", required=True, help="Output file path (with extension .nii.gz or .mhd)")
    args = parser.parse_args()

    volume = load_slices(args.input_dir, args.prefix, args.start, args.end)
    if args.format == "nii":
        save_nifti(volume, args.out)
    else:
        save_mhd(volume, args.out)

if __name__ == "__main__":
    main()
