from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "rberberi/qwen3-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

app = FastAPI()


class ReadingRequest(BaseModel):
    sign: str
    topic: str


@app.post("/api/reading")
def reading(req: ReadingRequest):
    # Same wording as the training data in train.py
    message = f"My sign is {req.sign}. Give me my {req.topic} horoscope for today."

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": message}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    text = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    # If the reading hit the length limit mid-sentence, cut back to the last full sentence
    end = max(text.rfind(c) for c in ".!?")
    if end != -1:
        text = text[:end + 1]
    return {"reading": text}


app.mount("/", StaticFiles(directory="static", html=True))
