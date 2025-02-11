import gradio as gr
from huggingface_hub import InferenceClient, login
from transformers import AutoModelForCausalLM, AutoTokenizer

# exemple : https://github.com/blancsw/deep_4_all/blob/main/tgi_demo/app.py

login("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")


client = InferenceClient("https://api-inference.huggingface.co/models/Gor-bepis/fact-checker-bfmtg-v1")


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

checkpoint = "Gor-bepis/fact-checker-bfmtg-v1"
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

def chat_local(message: str, history: list[dict]):
    history.append({"role": "user", "content": message})
    input_text = tokenizer.apply_chat_template(history, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)  
    outputs = model.generate(inputs, max_new_tokens=100, temperature=0.2, top_p=0.9, do_sample=True)
    decoded = tokenizer.decode(outputs[0])
    response = decoded.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]
    return response

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="teal"), title="BFMTG") as interface:
    gr.ChatInterface(
        fn=chat_local,
        type="messages",
        title="BFMTG",
        description="Bienvenue sur le chatbot qui vous dira si ce que vous avancez est vrai ou faux.",
        show_progress="full",
        examples=[
            "Is it true that the earth is flat ?", 
            "I've heard that the release of GTA VI will be postponed. I hope it's false, isn't it ?"
        ]
    )

interface.launch(debug=True)