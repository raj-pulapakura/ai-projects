from chatbot import Chatbot
from pathlib import Path

if __name__ == "__main__":
    index_path = (
        Path(__file__).resolve().parent / "index" / "flipkart_products_index.pkl"
    )
    chatbot = Chatbot(index_path=str(index_path))

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chatbot. Goodbye!")
            break
        response = chatbot.chat(user_input)
        print(f"\nChatbot:\n{response}\n")
