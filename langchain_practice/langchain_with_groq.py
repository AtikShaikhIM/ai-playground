import getpass
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

if __name__ == "__main__":
    load_dotenv()  # Load .env variables

    if not os.environ.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

    model = init_chat_model("llama3-8b-8192", model_provider="groq")

    messages = [
        SystemMessage("Act as Translator and help me translate language to Hindi on;y convert it to hindi no other explanation required"),
    ]

    while True:
        user_input = input("Human: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chat ended.")
            break

        messages.append(HumanMessage(user_input))
        response = model.invoke(messages)
        print("-----------------------AI---------------------- ")
        print("Response Content:", response.content)
        print("Tool Calls:", response.tool_calls)
        print("Invalid Tool Calls:", response.invalid_tool_calls)
        print("Usage Metadata:", response.usage_metadata)
        print("Response ID:", response.id)
        print("Response Metadata:", response.response_metadata)
        print("-----------------------AI---------------------- ")
        messages.append(AIMessage(response.content))
