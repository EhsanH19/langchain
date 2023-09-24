"""Test MongoDB Atlas Vector Search functionality."""
from __future__ import annotations

import os
from time import sleep
from typing import TYPE_CHECKING, Any

import pytest

from langchain.docstore.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch

from pymongo import MongoClient
from pymongo.collection import Collection

INDEX_NAME = os.environ.get("MONGODB_DB_INDEX_NAME")
CONNECTION_STRING = os.environ.get("MONGODB_ATLAS_URI")
DB_NAME = os.environ.get("MONGODB_DB_NAME")
COLLECTION_NAME = os.environ.get("MONGODB_COLLECTION_NAME")

# Instantiate as constant instead of pytest fixture to prevent needing to make multiple
# connections.


@pytest.fixture(scope="function", autouse=True)
def collection() -> Collection:
    test_client = MongoClient(CONNECTION_STRING)
    collection = test_client[DB_NAME][COLLECTION_NAME]
    yield collection
    collection.delete_many({})


def test_from_documents(collection: Collection, embedding_openai: Embeddings) -> None:
    """Test end to end construction and search."""
    documents = [
        Document(page_content="Dogs are tough.", metadata={"a": 1}),
        Document(page_content="Cats have fluff.", metadata={"b": 1}),
        Document(page_content="What is a sandwich?", metadata={"c": 1}),
        Document(page_content="That fence is purple.", metadata={"d": 1, "e": 2}),
    ]
    vectorstore = MongoDBAtlasVectorSearch.from_documents(
        documents,
        embedding_openai,
        collection=collection,
        index_name=INDEX_NAME,
    )
    sleep(1)  # waits for mongot to update Lucene's index
    output = vectorstore.similarity_search("Sandwich", k=1)
    assert output[0].page_content == "What is a sandwich?"
    assert output[0].metadata["c"] == 1


def test_from_texts(collection: Collection, embedding_openai: Embeddings) -> None:
    texts = [
        "Dogs are tough.",
        "Cats have fluff.",
        "What is a sandwich?",
        "That fence is purple.",
    ]
    vectorstore = MongoDBAtlasVectorSearch.from_texts(
        texts,
        embedding_openai,
        collection=collection,
        index_name=INDEX_NAME,
    )
    sleep(1)  # waits for mongot to update Lucene's index
    output = vectorstore.similarity_search("Sandwich", k=1)
    assert output[0].page_content == "What is a sandwich?"


def test_from_texts_with_metadatas(
    collection: Collection, embedding_openai: Embeddings
) -> None:
    texts = [
        "Dogs are tough.",
        "Cats have fluff.",
        "What is a sandwich?",
        "The fence is purple.",
    ]
    metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
    vectorstore = MongoDBAtlasVectorSearch.from_texts(
        texts,
        embedding_openai,
        metadatas=metadatas,
        collection=collection,
        index_name=INDEX_NAME,
    )
    sleep(1)  # waits for mongot to update Lucene's index
    output = vectorstore.similarity_search("Sandwich", k=1)
    assert output[0].page_content == "What is a sandwich?"
    assert output[0].metadata["c"] == 1


def test_from_texts_with_metadatas_and_pre_filter(
    collection: Collection, embedding_openai: Embeddings
) -> None:
    texts = [
        "Dogs are tough.",
        "Cats have fluff.",
        "What is a sandwich?",
        "The fence is purple.",
    ]
    metadatas = [{"a": 1}, {"b": 1}, {"c": 1}, {"d": 1, "e": 2}]
    vectorstore = MongoDBAtlasVectorSearch.from_texts(
        texts,
        embedding_openai,
        metadatas=metadatas,
        collection=collection,
        index_name=INDEX_NAME,
    )
    sleep(1)  # waits for mongot to update Lucene's index
    output = vectorstore.similarity_search(
        "Sandwich", k=1, pre_filter={"range": {"lte": 0, "path": "c"}}
    )
    assert output == []


def test_mmr(collection: Collection, embedding_openai: Embeddings) -> None:
    texts = ["foo", "foo", "fou", "foy"]
    vectorstore = MongoDBAtlasVectorSearch.from_texts(
        texts,
        embedding_openai,
        collection=collection,
        index_name=INDEX_NAME,
    )
    sleep(1)  # waits for mongot to update Lucene's index
    query = "foo"
    output = vectorstore.max_marginal_relevance_search(query, k=10, lambda_mult=0.1)
    assert len(output) == len(texts)
    assert output[0].page_content == "foo"
    assert output[1].page_content != "foo"
