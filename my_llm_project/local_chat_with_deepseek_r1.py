from langchain_ollama import OllamaLLM


"""

Make sure Ollama is running:
ollama serve

Make sure the model is available locally:
ollama pull deepseek-r1

Make sure langchain_ollama is installed:
pip install langchain_ollama

Check which models are downloaded:
curl http://localhost:11434/api/tags

"""

if __name__ == "__main__":

    model ="llama3.2"
    # model ="deepseek-r1"

    llm = OllamaLLM(model=model)

    print(f"Warming up the {model} model, please wait...")
    llm.invoke("Hello")  # Pre-warm with a quick dummy call
    print("✅ Model is ready! Type your question (or 'exit' to quit).")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("👋 Exiting. Have a great day!")
            break

        response = llm.invoke(user_input)
        print("LLM:", response)
