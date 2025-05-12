import time
from typing import Dict, Tuple

import numpy as np
from openai import AzureOpenAI

from call_analytics.settings import (
    AZURE_OPENAI_DEPLOYMENT_NAME,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
)


def authenticate_openai_client() -> AzureOpenAI:
    """
    Authenticate and return an AzureOpenAI client instance.

    This function initializes and returns an authenticated `AzureOpenAI` client
    using the provided API key, API version, and Azure endpoint.

    Returns:
    -------
    AzureOpenAI
        An authenticated AzureOpenAI client instance ready for use with Azure-hosted OpenAI models.

    Notes:
    -----
    - Requires the global variables `AZURE_OPENAI_KEY` and `AZURE_OPENAI_ENDPOINT` to be set.
    - Assumes that the `azure-openai` library is installed and imported correctly.
    """
    openaiClient = AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        api_version="2024-10-21",
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
    return openaiClient


def get_embeddings(
    client: AzureOpenAI, text: str, model: str = "text-embedding-3-large"
) -> Tuple[list[float], Dict[str, int]]:
    """
    Generate text embeddings using an Azure OpenAI embedding model.

    This function sends the input text to the specified Azure OpenAI embedding model
    and returns the resulting embedding vector along with the token usage. If the
    request fails, it retries once after a 1-second delay. If both attempts fail,
    it returns a zero vector of appropriate size based on the model and usage as 0.

    Parameters:
    ----------
    client : AzureOpenAI
        An authenticated AzureOpenAI client instance.

    text : str
        The input text to embed. Newlines are replaced with spaces before sending.

    model : str, default="text-embedding-3-large"
        The embedding model to use. Supported values include:
        - "text-embedding-3-large" (3072-dimensional output)
        - "text-embedding-ada-002" (1536-dimensional output)

    Returns:
    -------
    Tuple[list[float], int]
        A tuple containing:
        - The embedding vector as a list of floats.
        - The number of tokens used during the embedding request.

    Notes:
    -----
    - If both attempts fail, a zero vector is returned with length depending on the model.
    - This function prints a retry message on the first failure.
    - Consider handling specific exceptions for better error diagnostics.
    """
    input = text.replace("\n", " ")
    try:
        response = client.embeddings.create(input=[input], model=model)
        embeddings = response.data[0].embedding
        usage = response.usage.total_tokens
        return embeddings, usage
    except:
        time.sleep(1)
        print("Retrying")
        try:
            response = client.embeddings.create(input=[input], model=model)
            embeddings = response.data[0].embedding
            usage = response.usage.total_tokens
            return embeddings, usage
        except:
            if model == "text-embedding-3-large":
                embeddings = np.zeros(3072)
            elif model == "text-embedding-ada-002":
                embeddings = np.zeros(1536)
            return embeddings, 0


# Call Azure OpenAI for a single prompt
def get_guided_summary(client: AzureOpenAI, instructions: str, prompt: str) -> str:
    """
    Generate a guided summary using a chat-based Azure OpenAI model.

    This function sends a system instruction and a user prompt to the Azure OpenAI
    chat completion endpoint to generate a guided summary (or other structured output)
    based on the provided instructions.

    Parameters:
    ----------
    client : AzureOpenAI
        An authenticated AzureOpenAI client instance.

    instructions : str
        System-level instructions that guide the behavior of the model (e.g., format,
        tone, or specific extraction tasks).

    prompt : str
        The user-level prompt that contains the content to be summarized or processed.

    Returns:
    -------
    str
        The content of the model's response message. If the request fails, returns
        a fallback error string: "Error: no usable response".

    Notes:
    -----
    - The model deployment name is assumed to be configured in the AzureOpenAI client.
    - The function currently retries only once (implicitly via `try`), without delay.
    - You may customize the fallback behavior or implement retries for robustness.
    """
    try:
        response = client.chat.completions.create(
            # model=AZURE_OPENAI_DEPLOYMENT_NAME, # For some reason this doesn't work
            # model="chatbot-llm-poc",
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except:
        return "Error: no usable response"
