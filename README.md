---
title: Sunprint
emoji: ♐
colorFrom: blue
colorTo: indigo
sdk: static
app_file: static/index.html
pinned: false
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

# Sunprint

Daily horoscope readings from [rberberi/qwen3-finetuned](https://huggingface.co/rberberi/qwen3-finetuned), a Qwen3-0.6B model fine-tuned on daily horoscopes.

Pick your sign on the zodiac wheel and a topic (general, love, career or wellness) to get today's horoscope. The model runs in your browser on the GPU with [Transformers.js](https://huggingface.co/docs/transformers.js) and WebGPU, so there is no server: the first visit downloads the model (about 490 MB) and the browser caches it.

## Files

- `train.py` fine-tunes Qwen3-0.6B on [karthiksagarn/astro_horoscope](https://huggingface.co/datasets/karthiksagarn/astro_horoscope), following the [Transformers fine-tuning guide](https://huggingface.co/docs/transformers/main/training). Run it on a GPU, for example a Colab T4.
- `convert.py` turns the fine-tuned model into a 4-bit ONNX model for WebGPU with the onnxruntime-genai model builder and uploads it to [rberberi/qwen3-finetuned-onnx](https://huggingface.co/rberberi/qwen3-finetuned-onnx). Rerun it after retraining.
- `static/index.html` is the whole app.

## Run locally

Serve the `static` folder with any web server, for example:

```
python -m http.server 8000 --directory static
```

Then open http://localhost:8000.
