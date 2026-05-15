# Sparse, Revision-aware Interactive 3D Tumor Segmentation

PyTorch implementation accompanying the paper *"Sparse-slice Scribble
Training and Revision-aware Refinement for Interactive 3D Tumor
Segmentation"* (in submission).

The model extends [PRISM](https://arxiv.org/abs/2404.15028) with two
contributions:

1. **3-state revision-aware refine input** — a contradiction map next to
   the usual positive / negative click maps, so the refine head has a
   defined input state when a voxel is relabeled across rounds.
2. **Sparse-slice scribble training** — at each training step, scribbles
   are kept only on a few high-error slices along a randomly chosen
   anatomical axis, so the model learns to extrapolate 3D corrections
   from a small number of 2D anchor slices.

The training, inference, and stress-test scripts are kept compatible
with the original PRISM pipeline so prior results can be reproduced from
the same checkpoints by toggling CLI flags.

## Setup

```bash
git clone https://github.com/HaoLi12345/interactive_segmentation.git
cd interactive_segmentation
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Then install a torch build matching your CUDA / hardware, e.g.
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
```

## Data

Two public tumor benchmarks are supported.

* **KiTS21 kidney tumor** — `https://github.com/neheller/kits21`. Place
  one case per subdirectory:
  ```
  <data_root>/kits21/case_00000/imaging.nii.gz
  <data_root>/kits21/case_00000/aggregated_MAJ_seg.nii.gz
  <data_root>/kits21/split.pkl
  ```
* **MSD-Colon (Task10_Colon)** —
  `http://medicaldecathlon.com`. Layout:
  ```
  <data_root>/msd_colon/imagesTr/colon_<NNN>.nii.gz
  <data_root>/msd_colon/labelsTr/colon_<NNN>.nii.gz
  <data_root>/msd_colon/split.pkl
  ```

The accompanying `split.pkl` files are the verbatim PRISM 5-fold splits.
The dataloader expects `<data_root>/<dataset>/split.pkl`; see
`src/dataset/dataloader.py` for the exact format if you want to
regenerate them.

Pre-trained backbone weights (SAM ViT-B): download
`sam_vit_b_01ec64.pth` from
`https://github.com/facebookresearch/segment-anything` and place under
`./checkpoint_sam/sam_vit_b_01ec64.pth`.

## Train

Sparse-scribble revision-aware training on MSD-Colon:

```bash
python src/train.py \
    --data colon \
    --data_dir <data_root>/msd_colon \
    --save_dir ./outputs \
    --save_name ours \
    --iter_nums 11 --num_clicks 1 \
    --multiple_outputs --refine --use_box \
    --use_scribble --efficient_scribble \
    --use_3state_memory \
    --sparse_scribble_train --sparse_scribble_K_max 5 \
    --sparse_scribble_dense_prob 0.2 \
    --sparse_scribble_orientations axial,sagittal,coronal \
    --image_size 128 --batch_size 1 --num_workers 4 \
    --max_epoch 200 --lr 4e-5
```

KiTS21 is the same with `--data kits --data_dir <data_root>/kits21`.

To reproduce the PRISM baseline, omit `--use_3state_memory` and
`--sparse_scribble_*`; this matches the public PRISM training recipe
exactly.

## Inference

### Standard PRISM-ultra protocol

```bash
python src/test.py \
    --data colon \
    --data_dir <data_root>/msd_colon \
    --save_dir ./outputs --save_name ours --checkpoint best \
    --split test \
    --iter_nums 11 --num_clicks 1 --num_clicks_validation 10 \
    --multiple_outputs --refine --refine_test --use_box \
    --use_scribble --efficient_scribble \
    --use_3state_memory \
    --image_size 128 --batch_size 1 --num_workers 0
```

### Sparse scribbles (`K` informative slices)

Restrict the test-time scribble to the top-`K` residual-error slices
along a per-case random orientation:

```bash
python src/test.py ... \
    --test_K_slices 3 \
    --test_slice_orientation random
```

`K=1`, `K=3`, `K=5` reproduce the sparse-prompt results in the paper.

### Prompt-revision stress test

Inject relabeled prompts across rounds (`N` voxels per round):

```bash
python src/test.py ... \
    --test_K_slices 3 --test_slice_orientation random \
    --inter_iter_contradiction_N 100
```

Per-iteration Dice is emitted as `ITERDICE path=... iter=K dice=Y` log
lines; combined with `--save_per_iter_predictions` you also get the mask
nii.gz at every iteration under
`<save_test_dir>/per_iter_pred/<data>/<save_name>/<case_id>/`.

## Repository layout

```
src/
  config/         Argparse-based configuration (config_args.py).
  dataset/        Dataset + dataloader (loads <data_root>/<ds>/split.pkl).
  models/         3D ViT image encoder, prompt encoder, mask decoder + 3-state Refine head.
  processor/      trainer.py (Trainer), trainer_basic.py (base loop), tester.py, validater.py.
  utils/          Scribble helpers, boundary utilities, plotting.
  train.py        Training entry point.
  test.py         Evaluation entry point (standard / sparse / revision protocols).
requirements.txt  Python dependencies (install torch separately for your CUDA).
```

## Acknowledgements

The image encoder, prompt encoder, and base refine head are ported from
[MedICL-VU/PRISM](https://github.com/MedICL-VU/PRISM). The 3D SAM
backbone is initialized from
[facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything).

## Citation

```bibtex
@article{li2026sparse,
  title   = {Sparse-slice Scribble Training and Revision-aware Refinement for Interactive 3D Tumor Segmentation},
  author  = {Hao Li and {others}},
  journal = {in submission},
  year    = {2026},
}
```
