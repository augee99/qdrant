import boto3
import json
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter

# AWS Bedrock Configuration
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')  # Specify region and credentials if needed

# Qdrant Configuration
qdrant_client = QdrantClient(url="http://localhost:6333")  # Update with your Qdrant URL if needed

# Function to get embeddings from Amazon Bedrock
def get_text_embedding_from_bedrock(text):
    response = bedrock_client.invoke_model(
        modelId="amazon.titan-embed-text-v1",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text})
    )
    response_body = response['body'].read().decode('utf-8')
    embedding_data = json.loads(response_body)

    # Assuming the embedding vector is in the 'embedding' field in the response
    return embedding_data['embedding']

# Function to search for the nearest vectors in Qdrant
def search_qdrant(collection_name, query_embedding, top_k=1):
    results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k,  # Adjust the number of results (top-k)
        with_vectors=True  # Include vectors in the search results
    )
    return results

# Example query text to search
query_text = "At three o'clock precisely I was at Baker Street, but Holmes had not yet returned. The landlady informed me that he had left the house shortly after eight o'clock in the morning"  # Replace with your actual query

# 1. Get the query embedding using AWS Bedrock
query_embedding = get_text_embedding_from_bedrock(query_text)

# 2. Search the Qdrant collection for similar vectors
collection_name = "titan3_collection"  # Update with your actual collection name
top_k = 1  # Adjust the number of results to retrieve
search_results = search_qdrant(collection_name, query_embedding, top_k=top_k)

# 3. Process and display the search results
for result in search_results:
    #print(f"ID: {result.id}, Score: {result.score}, Payload: {result.payload}")
    print(f"ID: {result.id}, Score: {result.score}, Payload: {result.payload}, Vector: {result.vector}")
