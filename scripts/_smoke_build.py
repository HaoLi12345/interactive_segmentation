"""Build PRISM model on CPU and report trainable param counts. No forward pass."""
import sys
sys.path.insert(0, "/home/lih30/interactive_seg")
from src.config.loader import load_yaml
from src.models.build import build_model, count_trainable_params

cfg = load_yaml("/home/lih30/interactive_seg/configs/base/config.yaml")
print("dataset:", cfg.data.dataset)
print("patch_size:", cfg.data.patch_size)

model_dict = build_model(cfg, device="cpu")
print("loaded_sam:", model_dict["loaded_sam"])

counts = count_trainable_params(model_dict)
print("trainable params:")
for k, v in counts.items():
    print(f"  {k:20s} {v:>12,}")

print("MODEL BUILD OK")
