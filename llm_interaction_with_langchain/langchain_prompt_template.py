import getpass
import os
from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

def get_str_prompt_template_example(topic: str):
    template = "Tell me a joke about {topic}"
    prompt_template = PromptTemplate.from_template(template)
    prompt = prompt_template.format(topic=topic)  # Correctly format the prompt
    return prompt

def get_chat_prompt_template_example(topic: str):
    prompt_template = ChatPromptTemplate([
        ("system", "You are a helpful assistant"),
        ("user", "Tell me a joke about {topic}")
    ])
    prompt = prompt_template.format(topic=topic)  # Correctly format the prompt
    return prompt

def get_place_holder_prompt_template_example(messages: list):
    prompt_template = ChatPromptTemplate([
        ("system", "You are a helpful assistant"),
        MessagesPlaceholder("msgs")
    ])

    human_messages = [HumanMessage(content=message) for message in messages]
    prompt = prompt_template.format(msgs=human_messages)  # Correctly format the prompt
    return prompt

if __name__ == "__main__":
    if not os.environ.get("MISTRAL_API_KEY"):
        os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter API key for Mistral AI: ")

    model = init_chat_model("mistral-large-latest", model_provider="mistralai")

    try:
        # Example call for the string prompt template
        print("String Prompt Template Example:")
        str_template = get_str_prompt_template_example("Indian Politicians")
        response = model.invoke(str_template)
        print(response.content)

        # Example call for the chat prompt template
        print("\nChat Prompt Template Example:")
        chat_template = get_chat_prompt_template_example("Indian Kids")
        response = model.invoke(chat_template)
        print(response.content)

        # Example call for the messages placeholder
        print("\nMessages Placeholder Prompt Template Example:")
        placeholder_template = get_place_holder_prompt_template_example(["AsSalamuAlaikum!", "Provide me one verse from Quran which you like the most!"])
        response = model.invoke(placeholder_template)
        print(response.content)

    except Exception as e:
        print(f"An error occurred: {e}")
