import gradio as gr

# exemple : https://github.com/blancsw/deep_4_all/blob/main/tgi_demo/app.py


def chat(input: str, history: list[tuple[str, str]]):
    resp = "T'as gueule, " + input[::-1]
    input = input.lower()

    if input == "oui":
        resp = "non"

    # history.append({"role": "user", "content": input})
    # history.append({"role": "assistant", "content": resp})

    return resp

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