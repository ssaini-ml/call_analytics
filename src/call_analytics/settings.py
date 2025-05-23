import base64
import os
from typing import List

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

# Aircall API
AIRCALL_API_ID: str = os.getenv("AIRCALL_API_ID")
AIRCALL_API_TOKEN: str = os.getenv("AIRCALL_API_TOKEN")
AIRCALL_API_BASE64_CREDENTIALS: str = base64.b64encode(
    f"{AIRCALL_API_ID}:{AIRCALL_API_TOKEN}".encode()
).decode()
AIRCALL_API_URL: str = os.getenv("AIRCALL_API_URL")
AIRCALL_API_RATELIMIT: int = int(os.getenv("AIRCALL_API_RATELIMIT"))

# Aircall Numbers
AIRCALL_NUMBERS_CS: List[str] = os.getenv("AIRCALL_NUMBERS_CS").split(",")
AIRCALL_NUMBERS_PS: List[str] = os.getenv("AIRCALL_NUMBERS_PS").split(",")

# Azure Language
AZURE_LANGUAGE_ENDPOINT: str = os.getenv("AZURE_LANGUAGE_ENDPOINT")
AZURE_LANGUAGE_KEY: str = os.getenv("AZURE_LANGUAGE_KEY")
AZURE_LANGUAGE_ENTITIES_TO_REDACT: List[str] = os.getenv(
    "AZURE_LANGUAGE_ENTITIES_TO_REDACT"
).split(",")

AZURE_MAX_BATCH_SIZE: int = int(os.getenv("AZURE_MAX_BATCH_SIZE"))
AZURE_LANGUAGE_MAX_CHAR_LIMIT: int = int(os.getenv("AZURE_LANGUAGE_MAX_CHAR_LIMIT"))

# Azure OpenAI
AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY: str = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Local storage paths
PATH_AIRCALL_CALLS: str = os.getenv("PATH_AIRCALL_CALLS")
PATH_AIRCALL_DATA: str = os.getenv("PATH_AIRCALL_DATA")
PATH_AIRCALL_PROCESSED: str = os.getenv("PATH_AIRCALL_PROCESSED")
PATH_AIRCALL_SENTIMENTS: str = os.getenv("PATH_AIRCALL_SENTIMENTS")
PATH_AIRCALL_SUMMARIES: str = os.getenv("PATH_AIRCALL_SUMMARIES")
PATH_AIRCALL_TOPICS: str = os.getenv("PATH_AIRCALL_TOPICS")
PATH_AIRCALL_TRANSCRIPTIONS: str = os.getenv("PATH_AIRCALL_TRANSCRIPTIONS")
PATH_MLSTUDIO_LABELING_DATA: str = os.getenv("PATH_MLSTUDIO_LABELING_DATA")
PATH_MLSTUDIO_LABELED_DATA: str = os.getenv("PATH_MLSTUDIO_LABELED_DATA")
PATH_OPENAI_EMBEDDINGS: str = os.getenv("PATH_OPENAI_EMBEDDINGS")
PATH_FEATURES: str = os.getenv("PATH_FEATURES")
PATH_LABELED: str = os.getenv("PATH_LABELED")

RANDOM_STATE: int = int(os.getenv("RANDOM_STATE"))
