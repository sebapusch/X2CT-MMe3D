#!/bin/bash

set -e

REPO_URL="https://github.com/dek924/PerX2CT.git"
TARGET_DIR="../PerX2CT"
PATCH_DIR="patches"
CONFIG_DIR="configs"
CHECKPOINT_DIR="checkpoints"


# Clone original repo if missing
if [ ! -d "$TARGET_DIR" ]; then
  echo "Cloning PerX2CT into $TARGET_DIR..."
  git clone "$REPO_URL" "$TARGET_DIR"
else
  echo "PerX2CT already cloned at $TARGET_DIR."
fi

# Apply patches
echo "Applying patches"
cp "$PATCH_DIR/model.py" "$TARGET_DIR/x2ct_nerf/modules/inr/model.py"
cp "$PATCH_DIR/INREncoderZoomAxisInAlign.py" "$TARGET_DIR/x2ct_nerf/modules/INREncoderZoomAxisInAlign.py"

echo "Copying inference scripts"
cp "../inference.py" "$TARGET_DIR"
cp "../save_to_volume.py" "$TARGET_DIR"
cp "../generate_synthetic_volumes.py" "$TARGET_DIR"


# Copy config
echo "Copying config"
cp "../$CONFIG_DIR/PerX2CT.yaml" "$TARGET_DIR/$CONFIG_DIR/PerX2CT.yaml"

# Copy checkpoint
echo "Copying checkpoint"
mkdir "$TARGET_DIR/$CHECKPOINT_DIR"
cp "../$CHECKPOINT_DIR/PerX2CT.ckpt" "$TARGET_DIR/$CHECKPOINT_DIR/PerX2CT.ckpt"

echo "PerX2CT correctly set up at $TARGET_DIR"
