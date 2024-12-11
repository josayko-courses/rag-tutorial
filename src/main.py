import os

from dotenv import load_dotenv
from pymongo import MongoClient

from utils import load_dataset

load_dotenv()

# Initialize a MongoDB Python client
MONGODB_URI = os.getenv("MONGODB_URI")
mongodb_client = MongoClient(MONGODB_URI, appname="devrel.workshop.rag")
# Check the connection to the server
pong = mongodb_client.admin.command("ping")
if pong["ok"] == 1:
    print("MongoDB connection successful")

load_dataset()
