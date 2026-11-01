"""Copy the encoder of a T5 checkpoint (shared embedding + encoder.*) into a single safetensors file with the same
config and tokenizer, so that T5EncoderModel.from_pretrained loads identical weights from half the bytes.
Usage: python t5_encoder_only.py <model dir> <out dir>
"""
import glob
import os
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file

SRC, OUT = sys.argv[1:3]
os.makedirs(OUT, exist_ok=True)
tensors = {}
for f in sorted(glob.glob(f"{SRC}/model-*.safetensors")):
    with safe_open(f, framework="pt") as s:
        for k in s.keys():
            if k == "shared.weight" or k.startswith("encoder."):
                tensors[k] = s.get_tensor(k)
save_file(tensors, f"{OUT}/model.safetensors", metadata={"format": "pt"})
for f in ("config.json", "generation_config.json", "special_tokens_map.json", "spiece.model", "tokenizer.json",
          "tokenizer_config.json"):
    shutil.copy(f"{SRC}/{f}", OUT)
print(len(tensors), "tensors,", sum(t.numel() for t in tensors.values()) / 1e9, "B parameters")
