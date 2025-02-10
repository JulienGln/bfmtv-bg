import gradio as gr
from huggingface_hub import InferenceClient

# exemple : https://github.com/blancsw/deep_4_all/blob/main/tgi_demo/app.py

client = InferenceClient("https://api-inference.huggingface.co/models/Gor-bepis/fact-checker-bfmtg-v1", 
                         token="NOPE")


def chat(input: str, history: list[tuple[str, str]]):
    messages = []

    # gestion de l'historique
    for couple in history:
        messages.append({"role": couple["role"], "content": couple["content"]})

    # resp = input[::-1]
    
    messages.append({"role": "user", "content": input})

    chat_completion = client.chat.completions.create(
        model="Gor-bepis/fact-checker-bfmtg-v1",
        stream=True,
        messages=messages,
        max_tokens=1024
    )

    partial_message = ""
    for token in chat_completion:
        content = token.choices[0].delta.content
        if token.choices[0].finish_reason is not None:
            break
        partial_message += content
        yield partial_message

    # return resp

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="teal"), title="BFMTG") as interface:
    gr.ChatInterface(
        fn=chat,
        type="messages",
        title="BFMTG",
        description="Bienvenue sur le chatbot qui vous dira si ce que vous avancez est vrai ou faux.",
        show_progress="full",
        examples=[
            "Is it true that the earth is flat ?", 
            "I've heard that the release of GTA VI will be postponed. I hope it's false, isn't it ?"
        ]
    )

interface.launch()