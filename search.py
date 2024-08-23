import streamlit as st
import os
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

try:
    # Load environment variables
    load_dotenv()
    
    # Retrieve the MongoDB URI from environment variables
    MONGO_URI = os.getenv("MONGO_URI")
    
    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
    db = client["proprietary-database-investor"]  # Replace with your database name
    collection = db["company_embeddings"]
    investor_collection = db["data_source_1"]
    
    # Load the pre-trained sentence transformer model
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
except Exception as e:
        st.error(e)


def get_embedding(text):
    """Generates vector embeddings for the given text."""
    try:
        embedding = model.encode(text)
        return embedding.tolist()
    except Exception as e:
        st.error(e)


def vector_search(prompt, num_candidates, limit):
    try:
        # Generate the embedding for the prompt
        embedding = get_embedding(prompt)
        # Perform the vector search using MongoDB aggregation
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "company-description",
                    "path": "companyDescription_embedding",
                    "queryVector": embedding,
                    "numCandidates": num_candidates,
                    "limit": limit,
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "original_document_id": 1,
                    "company_name": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
    
        results = collection.aggregate(pipeline)
        return list(results)
    except Exception as e:
        st.error(e)


# Streamlit dashboard
st.title("Vector Search Dashboard")

# User input for prompt
prompt = st.text_input("Enter a prompt:", "")

if st.button("Search"):
    try:
            
        # Execute the vector search
        search_results = vector_search(prompt, num_candidates=10000, limit=100)
    
        # Process results to aggregate investor names
        aggregated_results = {}
        for result in search_results:
            company_name = result["company_name"]
            score = result["score"]
            document_id = result["original_document_id"]
    
            investor = investor_collection.find_one({"_id": document_id})
            investor_name = investor["investor"]
            company_description = investor["investmentDetails"][company_name][
                "companyDescription"
            ]
    
            if company_name not in aggregated_results:
                aggregated_results[company_name] = {
                    "score": score,
                    "description": company_description,
                    "investors": [],
                }
    
            aggregated_results[company_name]["investors"].append(investor_name)
    
        # Prepare data for displaying in a table
        table_data = []
        for company_name, data in aggregated_results.items():
            table_data.append(
                {
                    "Company Name": company_name,
                    "Score": data["score"],
                    "Company Description": data["description"],
                    "Investor Names": ", ".join(data["investors"]),
                }
            )
    
        # Display results in a table
        st.table(table_data)
    except Exception as e:
        st.error(e)
