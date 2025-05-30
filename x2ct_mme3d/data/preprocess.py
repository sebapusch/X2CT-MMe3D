from torchvision import transforms
import torchio

XRAY_TARGET_SIZE   = (512, 512)
VOLUME_TARGET_SIZE = (128, 128, 128)

def xray_pipeline() -> transforms.Compose:
    """Returns a transformation pipeline for X-ray images."""
    return transforms.Compose([
                    transforms.Resize(XRAY_TARGET_SIZE),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                ])
    
def ct_pipeline() -> torchio.Compose:
    """Returns a transformation pipeline for CT images."""
    return torchio.Compose([
        torchio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5)),
        torchio.ZNormalization(),
        torchio.Resample((1.0, 1.0, 1.0)),
        torchio.CropOrPad(VOLUME_TARGET_SIZE),
    ])