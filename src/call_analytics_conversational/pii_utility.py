"""
PII (Personally Identifiable Information) utility module for Call Analytics.
Provides functions for redacting PII from text using Azure Language Service.
"""

import asyncio

from azure.ai.textanalytics.aio import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

from .settings import (
    AZURE_LANGUAGE_DEFAULT_LANGUAGE,
    AZURE_LANGUAGE_ENDPOINT,
    AZURE_LANGUAGE_ENTITIES_TO_REDACT,
    AZURE_LANGUAGE_KEY,
    AZURE_LANGUAGE_MAX_CHAR_LIMIT,
    AZURE_MAX_BATCH_SIZE,
)


# Authenticate the async client
async def authenticate_pii_client() -> TextAnalyticsClient:
    credential = AzureKeyCredential(AZURE_LANGUAGE_KEY)
    return TextAnalyticsClient(endpoint=AZURE_LANGUAGE_ENDPOINT, credential=credential)


# Split a single long document into chunks within the character limit
def split_text_into_chunks(
    text: str, max_length: int = AZURE_LANGUAGE_MAX_CHAR_LIMIT
) -> list:
    if len(text) <= max_length:
        return [text]

    # Split on whitespace to avoid cutting words in half
    words = text.split()
    chunks, current_chunk = [], ""

    for word in words:
        if len(current_chunk) + len(word) + 1 <= max_length:
            current_chunk += " " + word if current_chunk else word
        else:
            chunks.append(current_chunk)
            current_chunk = word
    chunks.append(current_chunk)  # Add the last chunk

    return chunks


# Helper to prepare documents for PII processing and track original indices
def prepare_documents(documents: list) -> tuple:
    split_docs = []
    mapping = []  # Tracks (original_index, chunk_index) mapping

    for idx, doc in enumerate(documents):
        chunks = split_text_into_chunks(doc)
        split_docs.extend(chunks)
        mapping.extend([(idx, i) for i in range(len(chunks))])

    return split_docs, mapping


# Helper function to split into batches of specified size
def chunk_documents(documents: list, chunk_size: int = 5) -> list:
    return [documents[i : i + chunk_size] for i in range(0, len(documents), chunk_size)]


# Async function to process a single batch of documents
async def process_batch(client: TextAnalyticsClient, batch: list) -> list:
    response = await client.recognize_pii_entities(
        documents=batch, categories_filter=AZURE_LANGUAGE_ENTITIES_TO_REDACT
    )
    return [
        doc.redacted_text if not doc.is_error else f"Error: {doc.error.message}"
        for doc in response
    ]


# Redact PII with batching, splitting, and reconstruction
async def redact_pii_with_batches(client: TextAnalyticsClient, documents: list) -> list:
    split_docs, mapping = prepare_documents(documents)
    batches = chunk_documents(split_docs)

    async with client:
        tasks = [process_batch(client, batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks)

    # Flatten results
    redacted_chunks = [chunk for batch in batch_results for chunk in batch]

    # Reconstruct original documents
    reconstructed_docs = [""] * len(documents)
    chunk_groups = {}

    for (doc_idx, chunk_idx), redacted_chunk in zip(mapping, redacted_chunks):
        chunk_groups.setdefault(doc_idx, {})[chunk_idx] = redacted_chunk

    for doc_idx, chunks in chunk_groups.items():
        reconstructed_docs[doc_idx] = " ".join(
            [chunks[i] for i in sorted(chunks.keys())]
        )

    return reconstructed_docs
