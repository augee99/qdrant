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
def search_qdrant(collection_name, query_embedding, top_k=5):
    results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k,  # Adjust the number of results (top-k)
    )
    return results

# Example query text to search
#query_text = "At ten o'clock precisely I was at Baker Street, but Holmes had not yet returned. The landlady informed me that he had left the house shortly after eight o'clock in the morning"  # Replace with your actual query

query_text = "he asked , hoping his answer would be long and distracting . he obligingly described his current project . the company had bought up old boatyards along the harbour shore at balmain and luc was designing a new apartment complex to be built on the site . she listened to the pleasure and satisfaction in his voice as he explained what he wanted constructed and how it would take advantage of the view , as well as catering for every modern aspect of living in the city . clearly he enjoyed his work and the opportunity to have such lavish projects to work on . he might not recognise how deeply he was tied to the peretti corporation since it had always been there for him to step into , but skye did . big money at his fingertips . big money to invest how he saw fit . big money to spend how he pleased in his private life , as well . as long as he stayed where he belonged . or was that being unfair , too ? luc had more than enough driving forceto succeed in establishing himself anywhere , in any company , or on his own . why could n't she just accept that he did n't live in his father 's pocket ? because she could n't make the fear go away . it was too deeply rooted in past pain . 'do you still live at cronulla ? ' she asked , needing to know if he 'd continued living with his family in the incredibly luxurious horseshoe compound facing the waterfront there . he shook his head . 'dad sold that place five years ago . ' the timing made skye wonder if maurizio peretti had decided to shift his family right away from the neighbouring suburb of caringbah where luc 'sillegitimate child was possibly far too close for comfort . luc flashed her an ironic look . 'he upgraded to a heritage-listed mansion at bellevue hill"

# 1. Get the query embedding using AWS Bedrock
query_embedding = get_text_embedding_from_bedrock(query_text)

# 2. Search the Qdrant collection for similar vectors
collection_name = "titan3_collection"  # Update with your actual collection name
top_k = 2  # Adjust the number of results to retrieve
search_results = search_qdrant(collection_name, query_embedding, top_k=top_k)

# 3. Process and display the search results
for result in search_results:
    print(f"ID: {result.id}, Score: {result.score}, Payload: {result.payload}")
