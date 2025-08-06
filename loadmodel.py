# loadmodel.py
import ollama
import json

def select_model(prompt_text, models):
    """Prompt the user to select a model from a list."""
    print(prompt_text)
    for i, model in enumerate(models):
        # The key for the model name is 'model'
        print(f"{i + 1}: {model['model']}")
    
    while True:
        try:
            choice = int(input("Enter the number of your choice: "))
            if 1 <= choice <= len(models):
                return models[choice - 1]['model']
            else:
                print("Invalid number. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    """
    Main function to get local models, let user select, and save to config.
    """
    try:
        print("Fetching available Ollama models...")
        response = ollama.list()
        
        if 'models' not in response:
            print("Error: 'models' key not found in response from Ollama.")
            print("Full response:", response)
            return

        local_models = response['models']
        
        if not local_models:
            print("No local Ollama models found.")
            print("Please pull a model first, for example: ollama pull gemma:2b")
            return

        # Let user select the LLM
        llm_model_name = select_model(
            "\nPlease select a Large Language Model (LLM):",
            local_models
        )

        # Let user select the embedding model
        embedding_model_name = select_model(
            "\nPlease select an Embedding Model:",
            local_models
        )

        config = {
            "llm_model": llm_model_name,
            "embedding_model": embedding_model_name
        }

        with open('config.json', 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\nConfiguration saved to config.json:")
        print(json.dumps(config, indent=2))

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure the Ollama service is running.")

if __name__ == "__main__":
    main()