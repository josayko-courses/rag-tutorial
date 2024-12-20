import os
import time
from datetime import datetime
from typing import Any, List

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
history_collection = mongodb_driver.client[mongodb_driver.db_name]["chat_history"]
history_collection.create_index("session_id")


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


def store_chat_message(session_id: str, role: str, content: str) -> None:
    """
    Store a chat message in a MongoDB collection.

    Args:
        session_id (str): Session ID of the message.
        role (str): Role for the message. One of `system`, `user` or `assistant`.
        content (str): Content of the message.
    """
    # Create a message object with `session_id`, `role`, `content` and `timestamp` fields
    # `timestamp` should be set the current timestamp
    message = {
        "session_id": session_id,
        "role": role,
        "content": content,
        "timestamp": datetime.now(),
    }
    # Insert the `message` into the `history_collection` collection
    history_collection.insert_one(message)


def retrieve_session_history(session_id: str) -> List:
    """
    Retrieve chat message history for a particular session.

    Args:
        session_id (str): Session ID to retrieve chat message history for.

    Returns:
        List: List of chat messages.
    """
    # Query the `history_collection` collection for documents where the "session_id" field has the value of the input `session_id`
    # Sort the results in increasing order of the values in `timestamp` field
    cursor = history_collection.find({"session_id": session_id}).sort("timestamp", 1)

    if cursor:
        # Iterate through the cursor and extract the `role` and `content` field from each entry
        # Then format each entry as: {"role": <role_value>, "content": <content_value>}
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in cursor]
    else:
        # If cursor is empty, return an empty list
        messages = []

    return messages


def generate_answer_3(session_id: str, user_query: str) -> None:
    """
    Generate an answer to the user's query taking chat history into account.

    Args:
        session_id (str): Session ID to retrieve chat history for.
        user_query (str): The user's query string.
    """
    # Initialize list of messages to pass to the chat completion model
    messages = []

    # Retrieve documents relevant to the user query and convert them to a single string
    context = mongodb_driver.vector_search(COLLECTION_NAME, user_query)
    context = "\n\n".join([d.get("body", "") for d in context])
    # Create a system prompt containing the retrieved context
    system_message = {
        "role": "system",
        "content": f"Answer the question based only on the following context. If the context is empty, say I DON'T KNOW\n\nContext:\n{context}",
    }
    # Append the system prompt to the `messages` list
    messages.append(system_message)

    # Use the `retrieve_session_history` function to retrieve message history from MongoDB for the session ID `session_id`
    # And add all messages in the message history to the `messages` list
    message_history = retrieve_session_history(session_id)
    messages.extend(message_history)

    # Format the user message in the format {"role": <role_value>, "content": <content_value>}
    # The role value for user messages must be "user"
    # And append the user message to the `messages` list
    user_message = {"role": "user", "content": user_query}
    messages.append(user_message)

    # Call the chat completions API
    response = fw_client.chat.completions.create(model=model, messages=messages)

    # Extract the answer from the API response
    answer = response.choices[0].message.content  # type: ignore

    # Use the `store_chat_message` function to store the user message and also the generated answer in the message history collection
    # The role value for user messages is "user", and "assistant" for the generated answer
    store_chat_message(session_id, "user", user_query)
    store_chat_message(session_id, "assistant", answer)

    print(answer)


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
    # user_query_6 = "What is Atlas Search and how Atlas Search maps documents?"
    # print(user_query_6)
    # generate_answer_2(user_query_6)

    print("What are triggers in MongoDB Atlas?")
    generate_answer_3(
        session_id="3",
        user_query="What are triggers in MongoDB Atlas?",
    )

    time.sleep(3)

    print("What did I just ask you?")
    generate_answer_3(
        session_id="3",
        user_query="What did I just ask you?",
    )
