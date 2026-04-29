"""Run from project root: cd /home/lih30/interactive_seg && python -m src._check_imports"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
SRC = os.path.dirname(os.path.abspath(__file__))                    # src/
for p in (ROOT, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)
print("project root:", ROOT)

# PRISM internal mixed absolute/relative imports — works only when both `ROOT` and `SRC` on sys.path
from src.utils.util import setup_logger, _bbox_mask  # absolute style
from src.utils import scribble, boundary_selection
from src.config.config_args import parser, check_and_setup_parser
from src.models.build_sam3D import sam_model_registry3D
from src.dataset.dataloader import Dataset_promise
from src.processor.trainer import Trainer
from src.processor.tester import tester
print("PRISM imports OK")
