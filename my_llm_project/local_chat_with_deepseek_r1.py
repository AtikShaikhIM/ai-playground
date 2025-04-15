from langchain_ollama import OllamaLLM

"""
Simple Local Chatbot using Ollama + LangChain

💡 Prerequisites:
1. Start Ollama server:
    ollama serve

2. Ensure required models are downloaded:
    ollama pull deepseek-r1
    ollama pull llama3.2
    ollama pull mistral

3. Install the Python package:
    pip install langchain_ollama

4. Verify which models are available:
    curl http://localhost:11434/api/tags
"""

if __name__ == "__main__":
    # model = "mistral"
    model = "llama3.2"
    # model = "deepseek-r1"

    llm = OllamaLLM(model=model)

    print(f"\nWarming up the '{model}' model, please wait...")
    llm.invoke("Hello")  # Pre-warm with a simple message
    print(f"✅ '{model}' model is ready! Start chatting (type 'exit' to quit).\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting chat. See you next time!")
            break

        response = llm.invoke(user_input)
        print(f"{model.upper()}:", response, "\n")
