# rewriter.py
from transformers import pipeline

model = pipeline("text2text-generation", model="google/flan-t5-base")

def rewrite_sop(text):
    prompt = f"Improve this SOP: {text}"
    output = model(prompt, max_length=512, do_sample=True)
    return output[0]['generated_text']
