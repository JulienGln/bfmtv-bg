import gradio as gr
from huggingface_hub import InferenceClient, login
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# exemple : https://github.com/blancsw/deep_4_all/blob/main/tgi_demo/app.py

login("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")


client = InferenceClient()


def chat(input: str, history: list[tuple[str, str]]):
    messages = []

    # gestion de l'historique
    for couple in history:
        messages.append({"role": couple["role"], "content": couple["content"]})

    # resp = input[::-1]
    
    messages.append({"role": "user", "content": input})

    chat_completion = client.chat.completions.create(
        model="Gor-bepis/fact-checker-bfmtg-v2",
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

checkpoint = "Gor-bepis/fact-checker-bfmtg-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
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
            "TUCKER CARLSON Tells Liberal Guest: National Endowment for the Arts Is In Effect ‘Welfare for Rich, Liberal Elites’ [VIDEO] Tucker Carlson asks Robin Bronk, CEO of the Creative Coalition: Why should in a time of budge deficits, taxpayers be subsidizing entertainment for rich people ? Don t you think it s kind of funny that artists who are against the grain and thinking for themselves, all of a sudden they re queuing up for their handouts from taxpayers? Tucker asked. Why wouldn t artists just strike out on their own and be independent? Watch the back-and-forth here: FOX Insider",
            "Trump arrives in South Korea for talks on nukes, trade OSAN, South Korea (Reuters) - U.S. President Donald Trump landed in South Korea on Tuesday, the second leg of his 12-day Asia trip dominated by the North Korean nuclear standoff. South Koreans are bracing for the possibility that Trump s state visit could risk further inflaming tensions with North Korean leader Kim Jong Un, who has stepped up his pursuit of nuclear weapons that could soon be capable of striking the mainland United States. Trump will visit with U.S troops and is also expected to raise criticisms of a U.S-South Korean trade pact when he meets with President Moon Jae-in in Seoul.",
            "AWESOME! STREET ARTIST SABO TARGETS Hollywood Liberals With TRUMP ’24’ Spoof Posters Protesters gathered Friday in Los Angeles for something called United Against Hate Inauguration March, where street artist Sabo posted several fake advertisements for 24: Legacy. Sabo is our favorite conservative street artist famous for lampooning liberal Hollywood. He s celebrated the inauguration of Trump at an anti-Trump rally Friday in Los Angeles by posting faux posters of Fox s revival of the hit TV show 24:Read more: Hollywood Reporter"
        ]
    )

interface.launch(debug=True)