import random
import gradio as gr
from minbpe import RegexTokenizer
from typing import List
from pydantic import BaseModel


# Initialize the tokenizer
tokenizer = RegexTokenizer()
tokenizer.load("./models/regex.model")


class TokenInfo(BaseModel):
    token_id: int
    token_bytes: str
    token_text: str


class TokenRequest(BaseModel):
    text: str


class TokenResponse(BaseModel):
    token_ids: List[int]
    token_details: List[TokenInfo]
    full_text: str


def colorize_tokens(api_response):
    tokens = api_response["token_details"]

    colored_text = ""

    for token in tokens:
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        colored_text += f"<span style='color:{color}; font-size:20px; font-weight:bold;'>{token['token_text']}</span>"

    return colored_text


def colorize_tokens(tokens):
    colored_text = ""

    for token in tokens:
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))

        colored_text += f"""
        <span style="
            color:{color};
            font-size:20px;
            font-weight:bold;
            margin-right:6px;
        ">
            {token}
        </span>
        """

    return colored_text


# Mock function (replace with FastAPI call)
def process_input(text):
    token_ids = tokenizer.encode(text)

    # Get detailed information for each token
    token_details = []
    for token_id in token_ids:
        # Get the bytes for this token
        token_bytes = tokenizer.vocab[token_id]

        # Convert bytes to string representation, replacing invalid chars
        token_text = token_bytes.decode("utf-8",errors="replace")

        token_details.append(token_text)

    return colorize_tokens(token_details)


examples = [
    [
        "ஆதி அந்தமில்லாத கால வெள்ளத்தில் கற்பனை ஓடத்தில் ஏறி நம்முடன் சிறிது நேரம் பிரயாணம் செய்யுமாறு நேயர்களை அழைக்கிறோம்."
    ],
    ["ஆனந்த சிலை மனம் நெகிழ கண்டார்"],
    ["சிந்தாமணி, சிலப்பதிகாரம், மணிமேகலை,  வளையாபதி,  குண்டலகேசி  இவை  ஐம்பெருங்காப்பியமாம்"],
    ["சூளாமணி, யசோதர காவியம், உதயண  காவியம்,  நாககுமார  காவியம், நீலகேசி  இவை  ஐஞ்சிறுகாப்பியமாம்"],
]

with gr.Blocks(title="Tamil Tokenizers") as demo:
    gr.Markdown("# 🎨 Tokenizer Visualizer")
    gr.Markdown("Enter text to see tokenized output with color highlighting.")

    with gr.Row():
        inp = gr.Textbox(
            label="Input Text", placeholder="Type something in Tamil...", lines=2
        )

    with gr.Row():
        out = gr.HTML(label="Token Visualization", autoscroll=True)

    clear_btn = gr.Button("Clear")
    clear_btn.click(lambda: ("", ""), outputs=[inp, out])

    inp.change(process_input, inputs=inp, outputs=out)

    gr.Examples(examples=examples, inputs=inp)
if __name__ == "__main__":
    demo.launch(css="footer {visibility: hidden;}", theme=gr.themes.Monochrome())
