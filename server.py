from transformers import AutoTokenizer, AutoModel
from langchain_community.vectorstores import Pinecone
from embed import CustomEmbeddings
from flask import Flask, request
from dotenv import load_dotenv
from pinecone import Pinecone
from groq import Groq
import json
import requests
import os


load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
app = Flask(__name__)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MODEL = AutoModel.from_pretrained(MODEL_NAME)

custom_embeddings = CustomEmbeddings(MODEL, TOKENIZER)


with open("vids.json", encoding="utf-8") as f:
    videos = json.load(f)

index_name = "langchainvector"
index = pc.Index(index_name)

def get_response(question):
    query_embedding = custom_embeddings.embed_query(question)  # Call the method on the instance

    # Step 2: Perform similarity search in the Pinecone index
    results = index.query(
        namespace="",  # Replace with your namespace
        vector=query_embedding.tolist(),  # Convert to list if necessary
        top_k=7,  # Number of top similar results you want
        include_values=False,  # Optional: Exclude or include vector values in results
        include_metadata=True  # Optional: Include metadata in results
    )
    # print(results)
    combined_text = " ".join([match['metadata']['text'].strip() for match in results['matches']])

    # Prompt for Groq API to identify relevant headings
    prompt = f"""
        Answer the user's question in Spanish using only the provided text. Keep your response friendly, professional, and include emojis in the answer. If you find a relevant video from the provided list that can help strongly, recommend the url of it with the answer. Else, give answer without it. Do not mention the text source nor recommend unnecessary videos.
        Inputs:
        Question: {question}
        Text: {combined_text}
        Videos: {videos}
    """

    try:
        # Generate completion from Groq API
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "system",
                    "content": "You are responsible to find relevant headings that can help answer question",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )

        # Collect the response chunks
        extended_answer = ""
        for response_chunk in completion:
            extended_answer += response_chunk.choices[0].delta.content or ""

        return extended_answer

    except Exception as e:
        print(f"Error extending answer: {e}")
        return "no tengo datos sobre esta pregunta  :("
    


def send_message(res, chat_id):
    """Sends a message to a Telegram chat.
    
    Args:
        res (str): The message to be sent.
        chat_id (int): The chat ID to send the message to.
    """
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": res}
    requests.post(url=url, json=payload)


def handle_message(message):
    global USR_PLATFORM
    try:
        text: str = message["message"]["text"]
    except KeyError:
        # Handle the KeyError here, such as by assigning a default value to text
        print(message)
        print("No Text")
        return
    print(f"Received message: {text}")
    chat_id = message["message"]["chat"]["id"]
    response = get_response(text)
    print(f"Response: {response}")
    send_message(response, chat_id)

# abc
@app.route("/", methods=["GET", "POST"])
def bot_messages():
    if request.method == "POST":
        message = request.get_json()

        handle_message(message)

        return "OK", 200
    return "<h1>API ENDPOINT<h1>"

if __name__ == "__main__":
    app.run(debug=True, port=os.getenv("PORT"))