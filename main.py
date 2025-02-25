import gradio as gr
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from clients import LLMParamsDTO, LangchainGigaChat, LangchainOpenAI


with gr.Blocks(theme="CultriX/gradio-theme") as demo:
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(type="messages", render_markdown=True)
            msg = gr.Textbox(label="Введите сообщение")
            clear = gr.ClearButton([msg, chatbot], value="Очистить историю")

        with gr.Column(scale=1):
            model_choice = gr.Radio(
                choices=["GigaChat", "OpenAI"],
                value="GigaChat",
                label="Выберите модель",
            )
            temp = gr.Slider(0, 1, value=0.7, step=0.1, label="Temperature")
            max_tokens = gr.Slider(1028, 4068, value=1028, step=1, label="Max tokens")
            timeout = gr.Slider(5, 180, value=180, step=1, label="Timeout")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Ты ассистент. Любой текст присылаешь только в формате markdown",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )

    def get_chain(model_name: str, params):
        print(model_name)
        if model_name == "GigaChat":
            return prompt | LangchainGigaChat(params)
        else:
            return prompt | LangchainOpenAI(params)

    demo_ephemeral_chat_history_for_chain = ChatMessageHistory()

    def get_chain_with_history(chain):
        return RunnableWithMessageHistory(
            chain,
            lambda session_id: demo_ephemeral_chat_history_for_chain,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    def respond(
        message: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
        chat_history,
    ):
        params = LLMParamsDTO(
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        chat_history.append({"role": "user", "content": message})
        yield message, chat_history

        chain = get_chain(model_name, params)
        chain_with_message_history = get_chain_with_history(chain)

        bot_message = ""
        for chunk in chain_with_message_history.stream(
            {"input": message},
            {"configurable": {"session_id": chain_with_message_history}},
        ):
            bot_message += chunk.content

            if chat_history[-1]["role"] == "assistant":
                chat_history[-1]["content"] = bot_message
            else:
                chat_history.append({"role": "assistant", "content": bot_message})
            yield "", chat_history

    msg.submit(
        respond, [msg, model_choice, temp, max_tokens, timeout, chatbot], [msg, chatbot]
    )

if __name__ == "__main__":
    demo.launch()
