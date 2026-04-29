# interactive_seg

3D interactive medical-image segmentation with smarter prompting.

## Idea

Given a strong public 3D segmentation backbone, **smarter click selection** + **prompt curriculum borrowed from current SOTA** beats vanilla random-FP/FN click sampling under the same n-click protocol. The headline figure is *Dice vs #clicks* on KiTS21 and MSD-Colon, comparing PRISM, VISTA3D, nnInteractive, and ours.

- **Backbone:** PRISM architecture (selectively ported, modernized to PyTorch 2.4+).
- **Borrowed ideas (inference-only baselines):**
  - VISTA3D — three-state click (positive / negative / **ignore**), auto + click dual training mode
  - nnInteractive — multi-prompt curriculum (point → scribble → box → lasso), nnUNet-style data preprocessing
- **Our contribution:** click selection strategies (uncertainty-aware, boundary-aware, FN/FP top-1, expected-information-gain).
- **Datasets:** KiTS21 (kidney tumor), MSD-Colon (colon tumor). PRISM split files reused verbatim.

## Layout

| Path | Purpose |
|---|---|
| `src/` | Library code: `config/`, `data/`, `models/`, `prompts/`, `losses/`, `trainer/`, `evaluator/`, `metrics/`, `utils/` |
| `scripts/` | Entry points: `train_finetune.py`, `eval_zeroshot.py`, `eval_ours.py`, `benchmark_inference.py` |
| `configs/` | YAML configs: `base/` and `exp/<name>/` |
| `slurm/` | Submission scripts: `templates/{a6000,gh200}.slurm` (do not edit in place), `exp_*.slurm` |
| `docs/` | `handoff.md`, `decision_log.md`, `technical.md`, `dataset.md`, `borrowed_ideas.md` |
| `outputs/` | Checkpoints, logs (gitignored) |
| `results/` | Selected CSVs / figures for paper |
| `data/` | Symlinks only — never raw images |
| `splits/` | PRISM split.pkl files (committed; ~100 KB) |

## Conventions

Read `CLAUDE.md` and `AGENTS.md` before making changes. Key rules:

- All git-tracked files in English.
- No "Claude" anywhere in commit metadata or PR body.
- No data under `/home/lih30/` on ACCRE — use `/data/h_oguz_lab/lih30/`.
- VISTA3D and nnInteractive are inference-only; do not train.
- PRISM `split.pkl` files are reused verbatim.
- Smoke test verifies *numbers*, not "code didn't crash".

## Compute

ACCRE only (lab machine for code editing / data inspection):

```bash
sbatch slurm/templates/a6000.slurm   # accre_guests_acc + nvidia_rtx_a6000:1
sbatch slurm/templates/gh200.slurm   # h_oguz_lab_acc + nvidia_gh200:1 (preferred)
```

## Datasets

| Dataset | Source | Lab path | ACCRE path |
|---|---|---|---|
| KiTS21 | github.com/neheller/kits21 | `/media/hao/easystore/.../0SAM_data/kist_update/data/` | `/data/h_oguz_lab/lih30/interactive_seg_data/kits21/` |
| MSD-Colon | medicaldecathlon.com (Task10_Colon) | `/media/hao/easystore/.../0SAM_data/Task10_Colon/` | `/data/h_oguz_lab/lih30/interactive_seg_data/msd_colon/` |
| PRISM splits | — | `/media/hao/easystore/.../promise/datafile/{kits,colon}/split.pkl` | `splits/` (committed) |

Details in `docs/dataset.md`.
