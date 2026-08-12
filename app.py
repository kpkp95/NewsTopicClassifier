import os

import gradio as gr

from src.predict import NewsPredictor


predictor = NewsPredictor()


def classify_news(text: str):
    try:
        result = predictor.predict(text)
        label_text = f"{result['label']} ({result['confidence']:.1%} confidence)"
        return label_text, result["probabilities"]
    except ValueError as error:
        return str(error), {}


demo = gr.Interface(
    fn=classify_news,
    inputs=gr.Textbox(
        lines=7,
        label="News text",
        placeholder="Paste a news headline or short article here...",
    ),
    outputs=[
        gr.Textbox(label="Predicted category"),
        gr.Label(label="Class probabilities", num_top_classes=4),
    ],
    title="AG News Topic Classifier",
    description="DistilBERT classifies news as World, Sports, Business, or Sci/Tech.",
    examples=[
        ["The national football team won the championship after scoring in extra time."],
        ["The central bank announced a new interest-rate decision after its policy meeting."],
        ["Researchers introduced a faster processor designed for artificial intelligence systems."],
        ["World leaders met to discuss a new international peace agreement."],
    ],
)


if __name__ == "__main__":
    cloud_port = os.getenv("PORT")
    server_port = int(cloud_port) if cloud_port else 7860
    server_name = "0.0.0.0" if cloud_port else "127.0.0.1"
    demo.launch(server_name=server_name, server_port=server_port)
