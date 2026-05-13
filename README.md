# PASB: Pathology-Aware Schrödinger Bridge for Virtual IHC Staining

PASB is a weakly supervised virtual immunohistochemistry (IHC) staining framework.

PASB translates H&E images to virtual IHC images while using adjacent IHC sections as weak pathological guidance during training.

## Method Overview

PASB includes:

- **CDAL**: Constraint-Driven Alignment Learning. A pathology classifier aligns generated IHC images with high-level IRS/pathology labels from adjacent reference IHC sections.
- **SDPR**: Similarity-based Dynamic Path Refinement. During training bridge sampling, a similarity signal between generated IHC and adjacent reference IHC softly corrects the transport path and modulates noise.
- **PASB dataset mode**: Adjacent H&E/IHC pairs are read by matching index.

The main implementation files are:

```text
models/pasb_model.py      PASB model, losses, CDAL, training-time SDPR
data/pasb_dataset.py      weakly supervised adjacent-section dataset
run_pasb_train.sh         example PASB training command
run_pasb_test.sh          example PASB testing command
```

## Environment

Activate the project environment before training or testing:

```bash
mamba activate pasb
python -c "import torch; print(torch.__version__)"
```

The repository supports:

```bash
--gpu_ids -1     # CPU
--gpu_ids 0      # CUDA GPU
```

If `dominate` is missing:

```bash
python -m pip install dominate
```

## Data Layout

PASB expects adjacent H&E/IHC images under this folder layout:

```text
datasets/YourDataset/
  trainA/          H&E training patches
  trainB/          adjacent IHC reference patches
  testA/           H&E test patches
  labels.csv       optional pathology labels
```

`labels.csv` is optional. If provided, use:

```csv
patt,label
sample_000.png,0
sample_001.png,1
```

Labels are expected to be four IRS-like classes:

```text
0 negative
1 weak positive
2 moderate positive
3 strong positive
```

If `labels.csv` exists, PASB loads only the rows listed in the CSV and ignores unlabeled images. The `patt` value may be a basename, a path relative to `trainA`/`testA`, or a path relative to the dataset root. If no row can be matched to the current phase, dataset construction raises an error instead of falling back to pseudo labels.

`patt,label` is the recommended CSV schema. For compatibility, path columns may also be named `path`, `file`, `filename`, or `image`; label columns may also be named `y`, `irs_label`, or `pathology_label`. If neither a path column nor a label column is found, PASB prints a clear column-name error.

If labels are absent, PASB estimates a simple four-level pseudo prior from IHC color statistics for smoke tests. For real experiments, precomputed IRS/H-score-derived labels are recommended.

The PASB dataset applies the same random crop and flip parameters to each H&E/IHC pair so that weak adjacent-section correspondence is not destroyed by augmentation.

`testB` is not required for deployment. PASB inference uses H&E input only and does not run SDPR against a reference IHC image.

## Training

### Classifier Pretraining

For paper-faithful PASB reproduction, pretrain the pathology classifier on real IHC/IRS labels and freeze it during PASB training. `resnet50` is the default classifier backbone; a compact classifier is still available as `small`, and `resnet18` / `convnext_tiny` are also supported.

```bash
python train_pasb_classifier.py \
  --dataroot ./datasets/BCI \
  --name ihc_classifier \
  --image_dir trainB \
  --pasb_classifier_net resnet50 \
  --pasb_num_classes 4 \
  --batch_size 16 \
  --n_epochs 20 \
  --device auto
```

This saves `./checkpoints/ihc_classifier/latest_net_C.pth`, which PASB can load through `--pasb_pretrained_C`. Add `--pasb_classifier_pretrained_backbone` if you want torchvision ImageNet initialization for `resnet18`, `resnet50`, or `convnext_tiny`.

### PASB Training

By default the model uses `--pasb_classifier_net resnet50` and tries to load `./checkpoints/ihc_classifier/latest_net_C.pth`; if the file exists, `netC` is loaded and frozen.

```bash
python train.py \
  --dataroot ./datasets/BCI \
  --name bci_PASB \
  --model pasb \
  --mode pasb \
  --pasb_classifier_net resnet50 \
  --pasb_pretrained_C ./checkpoints/ihc_classifier/latest_net_C.pth \
  --lambda_SB 1.0 \
  --lambda_NCE 1.0 \
  --lambda_CDAL 1.0 \
  --batch_size 1 \
  --gpu_ids 0
```

If the classifier checkpoint is absent, the code falls back to online `netC` training so smoke tests and connectivity checks can still run. This fallback is only for engineering validation, not for final reproduction. For ablations, `--pasb_update_C true` also allows updating a loaded classifier during PASB training.

Or use:

```bash
bash run_pasb_train.sh
```

For small smoke tests, reduce the model width:

```bash
python train.py \
  --dataroot ./datasets/pasb_dummy_256 \
  --name pasb_smoke \
  --model pasb \
  --pasb_classifier_net small \
  --gpu_ids -1 \
  --ngf 4 \
  --ndf 4 \
  --load_size 256 \
  --crop_size 256 \
  --max_dataset_size 2 \
  --n_epochs 1 \
  --n_epochs_decay 0 \
  --num_patches 16 \
  --lr 0.00001 \
  --no_html
```

## Testing

Example:

```bash
python test.py \
  --dataroot ./datasets/BCI \
  --name bci_PASB \
  --model pasb \
  --mode pasb \
  --phase test \
  --epoch latest \
  --eval \
  --num_test 50 \
  --gpu_ids -1 \
  --checkpoints_dir ./checkpoints
```

Testing is H&E-only. SDPR is a training-time refinement and is not run during inference.

```bash
python test.py \
  --dataroot ./datasets/OnlyHE \
  --name bci_PASB \
  --model pasb \
  --mode pasb \
  --phase test \
  --epoch latest \
  --eval \
  --gpu_ids -1
```

Or use:

```bash
bash run_pasb_test.sh
```

Outputs are saved under:

```text
results/<experiment-name>/test_<epoch>/
```

Folders named `fake_1`, `fake_2`, ..., `fake_N` correspond to different NFE steps.

## Notes

- PASB is weakly supervised during training: `trainB` should contain adjacent IHC references for the corresponding H&E patches. Inference only requires `testA`.
- Use `--gpu_ids -1` for CPU smoke tests on Mac.
- The included `datasets/pasb_dummy` and `datasets/pasb_dummy_256` folders are synthetic smoke-test datasets only. They are not biologically meaningful.

## Citation

If you use this project, cite the PASB paper:

```bibtex
@article{qiu2026pasb,
  title={PASB: Pathology-aware Schrödinger bridge for virtual immunohistochemical staining},
  author={Qiu, Fanhao and Zhang, Yangyang and Huang, Zhen-Li and Zhu, Xiaofeng and Wang, Zhengxia},
  journal={Medical Image Analysis},
  volume={108},
  pages={103869},
  year={2026}
}
```
