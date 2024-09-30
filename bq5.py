import boto3
import json
import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.models import VectorParams

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

# Function to insert embedding into Qdrant with a valid ID
def insert_embedding_to_qdrant(collection_name, embedding, payload=None, vector_id=None):
    if vector_id is None:
        raise ValueError("vector_id is required when inserting into Qdrant")  # Ensure valid ID is provided

    point = PointStruct(
        id=vector_id,  # Ensure valid ID (unsigned integer or UUID)
        vector=embedding,
        payload=payload or {}
    )
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[point]
    )

# Create Qdrant collection if it doesn't exist
def create_qdrant_collection(collection_name, vector_size):
    try:
        # Check if collection already exists
        qdrant_client.get_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except:
        # If collection doesn't exist, create it
        print(f"Creating collection '{collection_name}'...")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance="Cosine")  # Adjust 'size' based on embedding dimensions
        )
        print(f"Collection '{collection_name}' created successfully.")

# Function to read the content from a file
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to chunk text into smaller parts
def chunk_text(text, chunk_size=500):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield ' '.join(words[i:i + chunk_size])

# Directory path containing the files
directory_path = 'data/'  # Update this with the actual directory path containing your files

# Get list of files in the directory (you can filter by extension if needed)
files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

# Process a maximum of 10 files
file_index=23
for file_index, file_name in enumerate(files[:10]):
    file_path = os.path.join(directory_path, file_name)

    # 1. Read the document from the file
    document_text = read_file(file_path)

    # 2. Chunk the document into smaller parts
    chunk_size = 500  # Adjust the size of chunks as needed
    chunks = list(chunk_text(document_text, chunk_size=chunk_size))

    # 3. Process each chunk
    collection_name = "titan4_collection"  # Update with your actual collection name in Qdrant
    embedding_size = 1536

    #create_qdrant_collection(collection_name, embedding_size)

    # Batch size for bulk insertion
    batch_size = 10  # You can adjust this number as needed

    # Store points in a list and insert them in batches
    points = []

    for idx, chunk in enumerate(chunks):
        embedding = get_text_embedding_from_bedrock(chunk)
        payload = {"file_name": file_name, "chunk_id": idx, "text": chunk}
        vector_id = file_index * 1000 + idx

        point = PointStruct(
            id=vector_id,
            vector=embedding,
            payload=payload
        )

        points.append(point)

        # Insert in batches of `batch_size`
        if len(points) >= batch_size:
            qdrant_client.upsert(collection_name=collection_name, points=points)
            print(f"Inserted a batch of {batch_size} points.")
            points = []  # Clear the batch after insertion

    # Insert remaining points (if any)
    if points:
        qdrant_client.upsert(collection_name=collection_name, points=points)
        print(f"Inserted the final batch of {len(points)} points.")

