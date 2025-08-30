from openai import OpenAI
from implementation import HNSW
from dotenv import load_dotenv
import json

load_dotenv()


class Chatbot:
    def __init__(self, index_path: str):
        self.client = OpenAI()
        self.index = HNSW.load(index_path)
        self.messages = []

    def chat(self, prompt: str) -> str:
        self.messages.append({"role": "user", "content": prompt})
        response = self.make_request()

        assistant_message = response.choices[0].message
        tool_call_results = []

        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                if tool_call.function.name == "search_index":
                    arguments = json.loads(tool_call.function.arguments)
                    query = arguments["query"]
                    query_embedding = self.get_embedding(query)
                    results = self.index.search(query_embedding, k=10)
                    tool_response = "\n".join(
                        f"Product: {result.metadata['product_name']}, Brand: {result.metadata['brand']}, Description: {result.metadata['description']}"
                        for result in results
                    )
                    tool_call_results.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_response,
                        }
                    )
            self.messages.append(assistant_message)
            self.messages.extend(tool_call_results)

            followup = self.make_request()
            return followup.choices[0].message.content
        else:
            return assistant_message.content

    def make_request(self):
        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=self.messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "search_index",
                        "description": "Search product index for relevant products",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query to find products",
                                },
                            },
                            "required": ["query"],
                        },
                    },
                }
            ],
        )

        return response

    def get_embedding(self, text: str) -> list:
        response = self.client.embeddings.create(
            input=text, model="text-embedding-3-small"
        )
        return response.data[0].embedding
