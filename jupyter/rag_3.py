from regex import X
import requests
import pandas as pd
import chromadb
from chromadb.api.types import QueryResult
from ollama import Client
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from datasets import Dataset

import requests
import chromadb
import re
import uuid


def parse_experiments():
    """ Get a list of secret experiment from ColdF  """
    url = "https://raw.githubusercontent.com/cgrodrigues/rag-intro/main/coldf_secret_experiments.txt"

    response = requests.get(url)
    if response.status_code == 200:
        text = response.text

        # Split the text using the experiment identifier as a delimiter
        experiments = text.split('# Experiment')
        
        # Remove empty strings and reformat each experiment
        experiments = ['# Experiment ' + exp.strip() for exp in experiments if exp.strip()]
        
        return experiments
    else:
        raise Exception(f"Failed to fetch the file: {response.status_code}")
    
def init_chroma_db(store_name:str="documents"):
    """ Initialize ChromaDB client. """
    chroma_client = chromadb.PersistentClient(path="./cromadb")
    vector_store = chroma_client.get_or_create_collection(name=store_name, metadata={"hnsw:space": "cosine"})
    return chroma_client, vector_store
    
def chunk_embed_text(input, get_embeddings, chunk_size:int=0, overlap_size:int=0 ):
    """Generate chunks and id from the list of texts."""

    chunks = []
    ids = []
    embeddings = []
    pattern = r"^# Experiment \d+.*"


    for text in input:
        start = 0
        
        if chunk_size == 0:
            _chunk_size = len(text) + overlap_size
        else:
            _chunk_size = chunk_size
        match = re.findall(pattern, text)
        if match:
            id = match[0]
        else: # some random id
            id = uuid.uuid4()
        ct = 0
        while start < len(text):
            # get the chunk
            end = start + _chunk_size
            chunk = f"{text[start:end]}"
            chunks.append(chunk)
            start += _chunk_size - overlap_size

            # get the embeddings
            # inputs = context_tokenizer(chunk, return_tensors='pt')
            # embedding = context_encoder(**inputs).pooler_output.detach().numpy()[0].tolist()
            embedding = get_embeddings.embeddings(chunk)
            embeddings.append(embedding)

            # get the id
            ids.append(f"{id}_{str(ct)}")
            ct += 1
            
    return chunks, ids, embeddings


def preprocess_text_to_chroma(text, get_embeddings, vector_store, chunk_size:int=0, overlap_size:int=0): 
    """Process text and store chunks in ChromaDB."""

    

  
    # Create the chunks, ids and embeddings from the experiment text to put in the database
    chunks, ids, embeddings = chunk_embed_text(input=text, 
                                               get_embeddings=get_embeddings, 
                                               chunk_size=chunk_size, 
                                               overlap_size=overlap_size)
    # Add to the database
    vector_store.add(documents=chunks, embeddings=embeddings, ids=ids)

def get_inference_prompt(question:str, get_embeddings, vector_store) -> tuple[str, QueryResult]:
    """ Based on the question get the most relevants chunks from the database and create the prompt to feed the model
        Return the prompt and the result of the search in the database"""
    
    query_embeddings = get_embeddings.embeddings(question)
    
     
    results = vector_store.query(query_embeddings, n_results=3)

    documents = "\n".join(results['documents'][0])

    prompt = f"""DOCUMENT:
{documents}

QUESTION:
{question}

INSTRUCTIONS:
Answer the user's QUESTION using the DOCUMENT markdown text above.
Provide short and concise answers.
Base your answer solely on the facts from the DOCUMENT.
If the DOCUMENT does not contain the necessary facts to answer the QUESTION, return 'NONE'."""

    return prompt, results


def get_inference(question, get_embeddings, vector_store):
    """ Inference in the LLaMA 3 model serve by Ollama """
    
    host = ""
    model = "llama3"
    prompt, db_results = get_inference_prompt(question,  get_embeddings, vector_store)

    system_message = {"role": "system", "content": prompt}
    messages = [system_message]

    response = Client(host=host).chat(model=model, messages=messages, options= {"seed": 42, "top_p": 0.9, "temperature": 0 })

    return response, prompt, db_results

def get_questions_answers_contexts(get_embeddings, store_name:str="documents"):
    """ Read the list of questions and answers and return a ragas dataset for evaluation """
    # URL of the file
    url = 'https://raw.githubusercontent.com/cgrodrigues/rag-intro/main/coldf_question_and_answer.psv'

    # Fetch the file from the URL
    response = requests.get(url)
    data = response.text

    # Get the Vector Database Client
    _, vector_store = init_chroma_db(store_name=store_name)

    # Split the data into lines
    lines = data.split('\n')

    # Split each line by the pipe symbol and create tuples
    rag_dataset = []

    for line in lines[1:]:
        if line.strip():  # Ensure the line is not empty
            question, reference_answer = line.split('|')
            response, _, db_results = get_inference(question,  get_embeddings, vector_store)

            generated_answer = response['message']['content']
            used_context = db_results['documents']

            rag_dataset.append({
                "question": question,
                "answer": generated_answer, 
                "contexts": [str(document) for document in used_context], 
                "ground_truth": reference_answer
            })

          
    rag_df = pd.DataFrame(rag_dataset)
    rag_eval_datset = Dataset.from_pandas(rag_df)
    
    # Return the lragas dataset
    return rag_eval_datset 