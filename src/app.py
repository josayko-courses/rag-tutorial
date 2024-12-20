import os
from typing import Any

from dotenv import load_dotenv
from fireworks.client import Fireworks
from sentence_transformers import CrossEncoder

from utils.mongo_driver import MongoDriver

load_dotenv()

# Initializing the Fireworks AI client and the model string
fw_client = Fireworks()
model = "accounts/fireworks/models/llama-v3-8b-instruct"

MONGODB_URI = os.getenv("MONGODB_URI")
assert isinstance(MONGODB_URI, str)
mongodb_driver = MongoDriver(MONGODB_URI)
mongodb_client = mongodb_driver.client
# Check the connection to the server
pong = mongodb_client.admin.command("ping")
if pong["ok"] == 1:
    print("MongoDB connection successful")

COLLECTION_NAME = "knowledge_base"


# Define a function to create the user prompt for our RAG application
def create_prompt(user_query: str) -> str:
    """
    Create a chat prompt that includes the user query and retrieved context.

    Args:
        user_query (str): The user's query string.

    Returns:
        str: The chat prompt string.
    """
    # Retrieve the most relevant documents for the `user_query` using the `vector_search` method
    context = mongodb_driver.vector_search(COLLECTION_NAME, user_query)
    # Join the retrieved documents into a single string, where each document is separated by two new lines ("\n\n")
    context = "\n\n".join([doc.get("body") for doc in context])
    # Prompt consisting of the question and relevant context to answer it
    prompt = f"Answer the question based only on the following context. If the context is empty, say I DON'T KNOW\n\nContext:\n{context}\n\nQuestion:{user_query}"
    return prompt


# Add a re-ranking step to the following function
def create_prompt_2(user_query: str) -> str:
    """
    Create a chat prompt that includes the user query and retrieved context.

    Args:
        user_query (str): The user's query string.

    Returns:
        str: The chat prompt string.
    """
    rerank_model = CrossEncoder("mixedbread-ai/mxbai-rerank-xsmall-v1")
    # Retrieve the most relevant documents for the `user_query` using the `vector_search` function defined in Step 8
    context = mongodb_driver.vector_search(COLLECTION_NAME, user_query)
    # Extract the "body" field from each document in `context`
    documents = [d.get("body") for d in context]
    # Use the `rerank_model` instantiated above to re-rank `documents`
    # Set the `top_k` argument to 5
    reranked_documents = rerank_model.rank(
        user_query, documents, return_documents=True, top_k=5
    )
    # Join the re-ranked documents into a single string, where each document is separated by two new lines ("\n\n")
    context = "\n\n".join([d.get("text", "") for d in reranked_documents])  # type: ignore
    # Prompt consisting of the question and relevant context to answer it
    prompt = f"Answer the question based only on the following context. If the context is empty, say I DON'T KNOW\n\nContext:\n{context}\n\nQuestion:{user_query}"
    return prompt


# Define a function to answer user queries using Fireworks' Chat Completion API
def generate_answer(user_query: str) -> None:
    """
    Generate an answer to the user query.

    Args:
        user_query (str): The user's query string.
    """
    # Use the `create_prompt` function above to create a chat prompt
    prompt = create_prompt_2(user_query)
    # Use the `prompt` created above to populate the `content` field in the chat message
    response: Any = fw_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    return response.choices[0].message.content


# Define a function to answer user queries in streaming mode using Fireworks' Chat Completion API
def generate_answer_2(user_query: str) -> None:
    """
    Generate an answer to the user query.

    Args:
        user_query (str): The user's query string.
    """
    # Use the `create_prompt` function defined in Step 9 to create a chat prompt
    prompt = create_prompt_2(user_query)
    # Use the `prompt` created above to populate the `content` field in the chat message
    # Set the `stream` parameter to True
    response = fw_client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], stream=True
    )

    # Iterate through the `response` generator and print the results as they are generated
    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")


if __name__ == "__main__":
    # Run the `generate_answer` function with a user query
    # user_query_1 = "What is MongoDB Atlas Search?"
    # answer_1 = generate_answer(user_query_1)
    # print(user_query_1)
    # time.sleep(3)
    # print(answer_1)
    # user_query_2 = "What did I just ask you ?"
    # answer_2 = generate_answer(user_query_1)
    # print(user_query_2)
    # time.sleep(3)
    # print(answer_2)
    # user_query_3 = "What time is it ?"
    # answer_3 = generate_answer(user_query_3)
    # print(user_query_3)
    # time.sleep(3)
    # print(answer_3)
    # user_query_4 = "How Atlas Search maps documents?"
    # answer_4 = generate_answer(user_query_4)
    # print(user_query_4)
    # time.sleep(3)
    # print(answer_4)
    # user_query_5 = "What are triggers in MongoDB Atlas?"
    # answer_5 = generate_answer(user_query_5)
    # print(user_query_5)
    # time.sleep(3)
    # print(answer_5)
    user_query_6 = "What is Atlas Search and how Atlas Search maps documents?"
    print(user_query_6)
    generate_answer_2(user_query_6)
