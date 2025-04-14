import os

from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

if __name__ == "__main__":
    if not os.environ.get("MISTRAL_API_KEY"):
        os.environ["MISTRAL_API_KEY"] = os.getenv("Enter API key for Mistral AI: ")

    model = init_chat_model("mistral-large-latest", model_provider="mistralai")

    messages = [
        SystemMessage(
            "Act as Translator and help me translate language to Hindi only convert it to asked language provide response in English level like My Name is Atik response should be Mera Naam Atik Hai no other explanation required"),
    ]

    response = model.invoke(messages)
    messages.append(AIMessage(response.content))

    while True:
        user_input = input("Human: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chat ended.")
            break

        messages.append(HumanMessage(user_input))
        response = model.invoke(messages)
        print("AI:", response.content)
        messages.append(AIMessage(response.content))