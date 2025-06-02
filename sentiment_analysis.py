import torch
from transformers import pipeline
import gradio as gr


DEVICE = "cuda:0"
MODEL_PATH = "distilbert-base-uncased-finetuned-sst-2-english"

distilbert_pipe = pipeline(
    model = MODEL_PATH,
    tokenizer = MODEL_PATH,
    device = DEVICE,
    top_k=None
)



def analyze_sentiment(text):
        result = distilbert_pipe(text,truncation=True, max_length=512, padding=True)
        y_pred = result[0][0].get('label')
        return f"Predicted sentiment: {y_pred}"



iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=[
        gr.Textbox(label="Input Text"),
    ],
    outputs=gr.Label(label="Sentiment Analysis Result"),
    title="Sentiment Analysis",
    description="Discover the emotional tone of any text with cutting-edge AI!"
)

iface.launch()