import os
import time
import pandas as pd
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from call_analytics.settings import (
    AZURE_LANGUAGE_KEY,
    AZURE_LANGUAGE_ENDPOINT,
    AZURE_LANGUAGE_ENTITIES_TO_REDACT,
    PATH_AIRCALL_DATA,
    PATH_AIRCALL_PROCESSED,
    PATH_AIRCALL_CALLS,
    PATH_AIRCALL_SUMMARIES,
    PATH_AIRCALL_TOPICS,
    PATH_AIRCALL_TRANSCRIPTIONS,
    PATH_AIRCALL_SENTIMENTS,
    AZURE_LANGUAGE_KEY,
    AZURE_LANGUAGE_MAX_CHAR_LIMIT,
)


# Authenticate the async client
async def authenticate_pii_client() -> TextAnalyticsClient:
    """
    Asynchronously authenticate and return an Azure Text Analytics client.

    This function initializes and returns an authenticated `TextAnalyticsClient`
    for use with Azure's Language service, such as PII (Personally Identifiable
    Information) detection.

    Returns:
    -------
    TextAnalyticsClient
        An authenticated client instance for calling Azure's Text Analytics APIs.

    Notes:
    -----
    - Requires global variables `AZURE_LANGUAGE_KEY` and `AZURE_LANGUAGE_ENDPOINT`
      to be set with valid credentials and endpoint URL.
    - Although marked as async, this function does not await anything internally.
      It is structured this way for consistency in async codebases.
    """
    credential = AzureKeyCredential(AZURE_LANGUAGE_KEY)
    return TextAnalyticsClient(endpoint=AZURE_LANGUAGE_ENDPOINT, credential=credential)


# Split a single long document into chunks within the character limit
def split_text_into_chunks(
    text: str, max_length: int = AZURE_LANGUAGE_MAX_CHAR_LIMIT
) -> list:
    """
    Split a long text into smaller chunks without cutting words, based on a max character limit.

    This function breaks a string into multiple chunks, each not exceeding `max_length` characters.
    It preserves word boundaries by splitting on whitespace. Useful for services (e.g., Azure)
    that enforce input character limits.

    Parameters:
    ----------
    text : str
        The input text to be split.

    max_length : int, default=AZURE_LANGUAGE_MAX_CHAR_LIMIT
        The maximum number of characters allowed per chunk.

    Returns:
    -------
    list of str
        A list of text chunks, each within the specified length limit.

    Notes:
    -----
    - Words are not broken across chunks; if a single word exceeds `max_length`, it will
      cause the chunk to exceed the limit.
    - Assumes that `AZURE_LANGUAGE_MAX_CHAR_LIMIT` is defined globally if not overridden.
    """
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
    """
    Split a list of documents into chunks and maintain a mapping to the original documents.

    This function takes a list of text documents, splits each into chunks using
    `split_text_into_chunks()`, and returns both the flattened list of all chunks and
    a mapping that relates each chunk back to its original document and chunk index.

    Parameters:
    ----------
    documents : list of str
        A list of text documents to be split into chunks.

    Returns:
    -------
    tuple:
        - split_docs (list of str): All resulting chunks from all documents.
        - mapping (list of tuple): Each tuple is (original_doc_index, chunk_index) indicating
          from which document and which chunk the entry in `split_docs` came from.

    Notes:
    -----
    - This is useful for processing documents in services with character limits while
      keeping track of which chunk came from which original document.
    - Relies on `split_text_into_chunks()` to handle chunking logic.
    """
    split_docs = []
    mapping = []  # Tracks (original_index, chunk_index) mapping

    for idx, doc in enumerate(documents):
        chunks = split_text_into_chunks(doc)
        split_docs.extend(chunks)
        mapping.extend([(idx, i) for i in range(len(chunks))])

    return split_docs, mapping


# Helper function to split into batches of specified size
def chunk_documents(documents: list, chunk_size: int = 5) -> list:
    """
    Split a list of documents into smaller batches (chunks) of a specified size.

    This function divides a list of documents into sublists, each containing up to
    `chunk_size` documents. Useful for batching requests to external services that
    have a limit on the number of items per request.

    Parameters:
    ----------
    documents : list
        The list of documents to be chunked.

    chunk_size : int, default=5
        The maximum number of documents per chunk.

    Returns:
    -------
    list of lists
        A list where each element is a sublist of documents, each of size up to `chunk_size`.

    Example:
    --------
    >>> chunk_documents(["doc1", "doc2", "doc3", "doc4"], chunk_size=2)
    [['doc1', 'doc2'], ['doc3', 'doc4']]
    """
    return [documents[i : i + chunk_size] for i in range(0, len(documents), chunk_size)]


# Async function to process a single batch of documents
async def process_batch(
    client: TextAnalyticsClient, batch: list, language: str
) -> list:
    """
    Asynchronously process a batch of documents for PII entity recognition and redaction.

    This function sends a batch of text documents to Azure's Text Analytics service
    to detect and redact Personally Identifiable Information (PII). It returns the
    redacted versions of the documents, or error messages if processing fails.

    Parameters:
    ----------
    client : TextAnalyticsClient
        An authenticated instance of Azure's Text Analytics client.

    batch : list of str
        A list of text documents to be processed.

    language : str
        The language code of the input documents (e.g., "en" for English, "nl" for Dutch).

    Returns:
    -------
    list of str
        A list containing the redacted version of each document.
        If a document processing fails, the corresponding entry will be an error message.

    Notes:
    -----
    - Uses the global constant `AZURE_LANGUAGE_ENTITIES_TO_REDACT` to filter specific
      PII entity categories.
    - Assumes the client is properly authenticated and supports the `recognize_pii_entities` API.
    - This function must be awaited in an async context.
    """
    response = await client.recognize_pii_entities(
        documents=batch,
        categories_filter=AZURE_LANGUAGE_ENTITIES_TO_REDACT,
        language=language,
    )
    return [
        doc.redacted_text if not doc.is_error else f"Error: {doc.error.message}"
        for doc in response
    ]


# Redact PII with batching, splitting, and reconstruction
async def redact_pii_with_batches(
    client: TextAnalyticsClient, documents: list, language: str
) -> list:
    """
    Asynchronously redact PII from a list of documents using batched processing.

    This function splits large documents into manageable chunks, processes them in
    parallel batches using Azure's Text Analytics PII recognition API, and then
    reconstructs the redacted documents.

    Parameters:
    ----------
    client : TextAnalyticsClient
        An authenticated instance of Azure's Text Analytics client.

    documents : list of str
        The input documents to redact for personally identifiable information (PII).

    language : str
        The language code of the documents (e.g., "en", "nl").

    Returns:
    -------
    list of str
        The redacted versions of the input documents, reconstructed from individually
        redacted chunks. If any chunk fails, it will contain an error message in place
        of the redacted content.

    Notes:
    -----
    - Documents are first split into chunks using `split_text_into_chunks()`.
    - Batches of chunks are created using `chunk_documents()`.
    - Processing is done concurrently using `asyncio.gather()` for performance.
    - Chunks are tracked with a mapping to correctly reassemble the redacted documents.
    - Uses the global `AZURE_LANGUAGE_ENTITIES_TO_REDACT` for filtering specific PII categories.
    - This function should be run in an async context.
    """
    split_docs, mapping = prepare_documents(documents)
    batches = chunk_documents(split_docs)

    async with client:
        tasks = [process_batch(client, batch, language) for batch in batches]
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
