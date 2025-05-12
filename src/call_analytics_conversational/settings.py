import base64
import os
from typing import List

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

# Aircall API
AIRCALL_API_ID: str = os.getenv("AIRCALL_API_ID", "")
AIRCALL_API_TOKEN: str = os.getenv("AIRCALL_API_TOKEN", "")
AIRCALL_API_BASE64_CREDENTIALS: str = base64.b64encode(
    f"{AIRCALL_API_ID}:{AIRCALL_API_TOKEN}".encode()
).decode()
AIRCALL_API_URL: str = os.getenv("AIRCALL_API_URL", "https://api.aircall.io/v1")
AIRCALL_API_RATELIMIT: int = int(os.getenv("AIRCALL_API_RATELIMIT", "50"))

# Aircall Numbers
AIRCALL_NUMBERS_CS: List[str] = os.getenv("AIRCALL_NUMBERS_CS", "").split(",")
AIRCALL_NUMBERS_PS: List[str] = os.getenv("AIRCALL_NUMBERS_PS", "").split(",")

# Azure Language Service
AZURE_LANGUAGE_ENDPOINT: str = os.getenv("AZURE_LANGUAGE_ENDPOINT", "")
AZURE_LANGUAGE_KEY: str = os.getenv("AZURE_LANGUAGE_KEY", "")
AZURE_LANGUAGE_API_VERSION: str = os.getenv("AZURE_LANGUAGE_API_VERSION", "2023-04-01")
AZURE_LANGUAGE_DEFAULT_LANGUAGE: str = os.getenv("AZURE_LANGUAGE_DEFAULT_LANGUAGE", "en")
AZURE_LANGUAGE_ENTITIES_TO_REDACT: List[str] = os.getenv(
    "AZURE_LANGUAGE_ENTITIES_TO_REDACT",
    "Person,Address,PhoneNumber,Organization,Email,Date,DateTime"
).split(",")

AZURE_MAX_BATCH_SIZE: int = int(os.getenv("AZURE_MAX_BATCH_SIZE", "5"))
AZURE_LANGUAGE_MAX_CHAR_LIMIT: int = int(os.getenv("AZURE_LANGUAGE_MAX_CHAR_LIMIT", "5000"))

# Azure OpenAI
AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_KEY: str = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_DEPLOYMENT_NAME: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")

# Local storage paths
PATH_AIRCALL_CALLS: str = os.getenv("PATH_AIRCALL_CALLS", "data/aircall/calls")
PATH_AIRCALL_DATA: str = os.getenv("PATH_AIRCALL_DATA", "data/aircall")
PATH_AIRCALL_PROCESSED: str = os.getenv("PATH_AIRCALL_PROCESSED", "data/aircall/processed")
PATH_AIRCALL_SENTIMENTS: str = os.getenv("PATH_AIRCALL_SENTIMENTS", "data/aircall/sentiments")
PATH_AIRCALL_SUMMARIES: str = os.getenv("PATH_AIRCALL_SUMMARIES", "data/aircall/summaries")
PATH_AIRCALL_TOPICS: str = os.getenv("PATH_AIRCALL_TOPICS", "data/aircall/topics")
PATH_AIRCALL_TRANSCRIPTIONS: str = os.getenv("PATH_AIRCALL_TRANSCRIPTIONS", "data/aircall/transcriptions")
PATH_MLSTUDIO_LABELING_DATA: str = os.getenv("PATH_MLSTUDIO_LABELING_DATA", "data/mlstudio/tolabel")

# Utility
RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", "42"))
