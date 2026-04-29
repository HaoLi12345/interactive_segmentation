#!/bin/bash
cd /home/lih30/interactive_seg/src
python -c "
from utils.util import setup_logger, _bbox_mask
from utils import scribble, boundary_selection
from config.config_args import parser, check_and_setup_parser
from models.build_sam3D import sam_model_registry3D
from dataset.dataloader import Dataset_promise
from processor.trainer import Trainer
from processor.tester import tester
print(\"PRISM imports OK\")
"
