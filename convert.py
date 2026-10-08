# Converts the fine-tuned model into a compressed ONNX model that runs in the browser with Transformers.js.
# Run it in a separate environment, because optimum-onnx needs an older transformers:
#   python -m venv .venv-convert
#   .venv-convert\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
#   .venv-convert\Scripts\pip install "optimum-onnx[onnxruntime]" onnx onnx-ir
#   .venv-convert\Scripts\python convert.py
import os
import shutil

import numpy as np
import onnx
from huggingface_hub import HfApi, hf_hub_download
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from optimum.exporters.onnx import main_export

model_name = "rberberi/qwen3-finetuned"
web_repo = "rberberi/qwen3-finetuned-onnx"
embeddings = "/model/embed_tokens/Gather"


def quantize_embeddings_8bit(model):
    """Store the word embeddings as int8 with one scale per row, using only ops every browser backend supports."""
    graph = model.graph
    node = next(n for n in graph.node if n.name == embeddings)
    weight = next(i for i in graph.initializer if i.name == node.input[0])
    w = numpy_helper.to_array(weight)
    scale = (np.abs(w).max(axis=1, keepdims=True) / 127).astype(np.float32)
    graph.initializer.remove(weight)
    graph.initializer.extend([
        numpy_helper.from_array(np.round(w / scale).astype(np.int8), "embed_q"),
        numpy_helper.from_array(scale, "embed_scale"),
    ])
    ids, out = node.input[1], node.output[0]
    position = list(graph.node).index(node)
    graph.node.remove(node)
    for i, new in enumerate([
        helper.make_node("Gather", ["embed_q", ids], ["embed_rows"], axis=0),
        helper.make_node("Cast", ["embed_rows"], ["embed_rows_f"], to=TensorProto.FLOAT),
        helper.make_node("Gather", ["embed_scale", ids], ["embed_row_scale"], axis=0),
        helper.make_node("Mul", ["embed_rows_f", "embed_row_scale"], [out]),
    ]):
        graph.node.insert(position + i, new)


# 1. Export to ONNX (full precision, about 3 GB)
main_export(model_name, output="onnx-export", task="text-generation-with-past")

# 2. Compress every weight to 8 bits (about 790 MB). 4 bits would be smaller, but it
#    noticeably hurt this small model: perplexity on real horoscopes went from 7.5 to 8.2-10.6
model = onnx.load("onnx-export/model.onnx")
quantize_embeddings_8bit(model)
quantizer = MatMulNBitsQuantizer(model, bits=8, block_size=32, is_symmetric=True, op_types_to_quantize=("MatMul",))
quantizer.process()
os.makedirs("onnx-web/onnx", exist_ok=True)
quantizer.model.save_model_to_file("onnx-web/onnx/model_quantized.onnx", use_external_data_format=False)

# 3. Add the config and tokenizer files Transformers.js needs, then upload
for name in ["config.json", "generation_config.json"]:
    shutil.copy(f"onnx-export/{name}", f"onnx-web/{name}")
for name in ["tokenizer.json", "tokenizer_config.json"]:
    shutil.copy(hf_hub_download(model_name, name), f"onnx-web/{name}")

api = HfApi()
api.create_repo(web_repo, exist_ok=True)
api.upload_folder(folder_path="onnx-web", repo_id=web_repo, commit_message="Upload compressed ONNX model")
print(f"Uploaded to https://huggingface.co/{web_repo}")
