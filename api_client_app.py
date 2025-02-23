# this is an app similar to app.py that connects to an api instead of runing the model on the same machine
import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient(base_url="http://localhost:8083")

def chat(input: str, history: list[tuple[str, str]]):
    messages = []

    # gestion de l'historique
    for couple in history:
        messages.append({"role": couple["role"], "content": couple["content"]})
    
    messages.append({"role": "user", "content": input})

    chat_completion = client.chat.completions.create(
        model="tgi",
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



with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="teal"), title="BFMTG") as interface:
    gr.ChatInterface(
        fn=chat,
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