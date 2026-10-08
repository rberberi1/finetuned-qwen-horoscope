# Converts the fine-tuned model into a compressed ONNX model that runs in the browser with Transformers.js (WebGPU).
# Uses the onnxruntime-genai model builder, which fuses attention into a few large ops; that made generation
# about 8x faster in the browser than a plain ONNX export. Run it in a separate environment:
#   python -m venv .venv-convert
#   .venv-convert\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
#   .venv-convert\Scripts\pip install onnxruntime-genai transformers onnx
#   .venv-convert\Scripts\python convert.py
import os
import runpy
import shutil
import sys

import onnx
import onnxruntime_genai.models
from huggingface_hub import HfApi, hf_hub_download

model_name = "rberberi/qwen3-finetuned"
web_repo = "rberberi/qwen3-finetuned-onnx"

# 1. Build a 4-bit model for WebGPU (about 490 MB); k_quant_mixed keeps the most sensitive layers at 8 bits
sys.path.insert(0, os.path.dirname(onnxruntime_genai.models.__file__))
import builders.base  # noqa: E402

# Keep rotary embeddings as a separate op with a position_ids input, the layout Transformers.js runs
builders.base.Model.is_fused_rope_supported = lambda self: False
sys.argv = ["builder", "-m", model_name, "-o", "onnx-build", "-p", "int4", "-e", "webgpu",
            "-c", "onnx-build/cache", "--extra_options", "algo_config=k_quant_mixed"]
runpy.run_module("onnxruntime_genai.models.builder", run_name="__main__")

# 2. Save it as one file in the layout Transformers.js expects (dtype "q4f16")
model = onnx.load("onnx-build/model.onnx")
for tensor in list(model.graph.input) + list(model.graph.output):
    for dim in tensor.type.tensor_type.shape.dim:
        if dim.dim_param == "kv_cache_dim":
            dim.dim_value = 128  # Transformers.js needs the KV cache head size as a number
os.makedirs("onnx-web/onnx", exist_ok=True)
onnx.save(model, "onnx-web/onnx/model_q4f16.onnx")
for name in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"]:
    shutil.copy(hf_hub_download(model_name, name), f"onnx-web/{name}")

# 3. Upload
api = HfApi()
api.create_repo(web_repo, exist_ok=True)
api.upload_folder(folder_path="onnx-web", repo_id=web_repo, commit_message="Upload WebGPU model", delete_patterns=["onnx/*"])
print(f"Uploaded to https://huggingface.co/{web_repo}")
