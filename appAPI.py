# appAPI.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

def get_groq_llm():
    """
    Initializes and returns a ChatGroq instance.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file")
    return ChatGroq(api_key=api_key)

def get_google_llm():
    """
    Initializes and returns a ChatGoogleGenerativeAI instance.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file")
    return ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

def get_openrouter_llm(model_name="google/gemini-pro"):
    """
    Initializes and returns a ChatOpenAI instance configured for OpenRouter.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file")
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

def main():
    """
    Main function to test the LLM providers.
    """
    print("Testing LLM providers...")

    try:
        print("\n--- Testing Groq ---")
        groq_llm = get_groq_llm()
        response = groq_llm.invoke("Explain the importance of low-latency LLMs")
        print("Groq response:", response.content)
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred with Groq: {e}")

    try:
        print("\n--- Testing Google ---")
        google_llm = get_google_llm()
        response = google_llm.invoke("What is the future of AI?")
        print("Google response:", response.content)
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred with Google: {e}")

    try:
        print("\n--- Testing OpenRouter ---")
        openrouter_llm = get_openrouter_llm()
        response = openrouter_llm.invoke("What are the main AI safety concerns?")
        print("OpenRouter response:", response.content)
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred with OpenRouter: {e}")

if __name__ == "__main__":
    main()