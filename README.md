---
title: Sunprint
emoji: ♐
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Sunprint

Daily horoscope readings from [rberberi/qwen3-finetuned](https://huggingface.co/rberberi/qwen3-finetuned), a Qwen3-0.6B model fine-tuned on 22,000 daily horoscopes.

Pick your sign on the zodiac wheel and a topic (general, love, career or wellness) to get today's horoscope.

## Files

- `train.py` fine-tunes Qwen3-0.6B on [karthiksagarn/astro_horoscope](https://huggingface.co/datasets/karthiksagarn/astro_horoscope), following the [Transformers fine-tuning guide](https://huggingface.co/docs/transformers/main/training). Run it on a GPU, for example a Colab T4.
- `app.py` is a FastAPI backend that loads the fine-tuned model and returns readings.
- `static/index.html` is the whole frontend.

## Run locally

```
pip install -r requirements.txt
python -m uvicorn app:app --port 8000
```

Then open http://localhost:8000. Each reading takes about 30 seconds on a CPU.
