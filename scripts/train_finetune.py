"""Entry point: finetune PRISM-style model on KiTS21 / MSD-Colon.

Usage:
    python scripts/train_finetune.py --config configs/exp/<name>/config.yaml [--resume <ckpt>]

Run only via SLURM (R-Process.LoginNode); do not invoke on the login node.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.loader import parse_config_args
from src.trainer import Trainer
from src.utils.logger import get_logger


def main() -> int:
    cfg, args = parse_config_args(description="Finetune interactive_seg model")
    log = get_logger("train", log_file=Path(cfg.experiment.output_dir) / "logs" / "train.log")
    log.info("config: %s", cfg)

    trainer = Trainer(cfg, logger=log)
    if hasattr(args, "resume") and args.resume:
        trainer.resume(args.resume)
    trainer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
