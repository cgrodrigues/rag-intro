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
    vector_store = chroma_client.get_or_create_collection(store_name)
    return chroma_client, vector_store
    
def chunk_embed_text(input, context_encoder, context_tokenizer):
    """Generate chunks and id from the list of texts."""

    chunks = []
    ids = []
    embeddings = []
    pattern = r"^# Experiment \d+.*"
    # Define a list of abbreviations, initials and titles to handle
    titles = ["Dr", "Mr", "Mrs", "Ms", "Prof", "Sr", "Jr", "St", "Ph.D", "M.D", "B.A", "M.A"]
    abbreviations = r'\b(?:{})\.\s+'.format('|'.join(titles))
    initials = r'\b[A-Z]\.\s+'


    for text in input:
        # Pre-process text to protect abbreviations, initials and titles from being split
        protected_text = re.sub(abbreviations, lambda x: x.group().replace('. ', '<dot> '), text)
        protected_text = re.sub(initials, lambda x: x.group().replace('. ', '<dot> '), protected_text)
            
        # Split text into sentences
        phrases = re.split(r'(?<!<dot>)[.!?]\s+', protected_text)

        # Restore protected abbreviations and initials
        phrases = [f"{phrase.replace('<dot>', '.')}." for phrase in phrases]
        match = re.findall(pattern, text)
        if match:
            id = match[0]
        else:
            id = uuid.uuid4()
        
        for ct, phrase in enumerate(phrases):
            # Ensure chunk size and overlap are not used in this implementation
            if ct == 0:
                chunk = phrase
            else:
                chunk = f"{id}\n\n{phrase}\n\n"
            chunks.append(chunk)
            
            # Get the embeddings
            inputs = context_tokenizer(chunk, return_tensors='pt')
            embedding = context_encoder(**inputs).pooler_output.detach().numpy()[0].tolist()
            embeddings.append(embedding)
            
            # Get the id
            ids.append(f"{id}_{str(ct)}")
            
    return chunks, ids, embeddings


def preprocess_text_to_chroma(text, vector_store): 
    """Process text and store chunks in ChromaDB."""

    # Get the encoder and tokenizer
    context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    
    # Create the chunks, ids and embeddings from the experiment text to put in the database
    chunks, ids, embeddings = chunk_embed_text(input=text, 
                                               context_encoder=context_encoder, 
                                               context_tokenizer=context_tokenizer)
    # Add to the database
    vector_store.add(documents=chunks, embeddings=embeddings, ids=ids)

def get_inference_prompt(question:str, context_encoder, context_tokenizer, vector_store) -> tuple[str, QueryResult]:
    """ Based on the question get the most relevants chunks from teh database and create the prompt to feed the model
        Return the prompt and the result of the search in the database"""
    
    inputs = context_tokenizer(question, return_tensors='pt')
    query_embeddings = context_encoder(**inputs).pooler_output.detach().numpy()[0].tolist()
     
    results = vector_store.query(query_embeddings, n_results=3)
    # results = vector_store.query(query_texts=question, n_results=10)

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


def get_inference(question, context_encoder, context_tokenizer, vector_store):
    """ Inference in the LLaMA 3 model serve by Ollama """
    
    host = ""
    model = "llama3"
    prompt, db_results = get_inference_prompt(question, context_encoder, context_tokenizer, vector_store)

    system_message = {"role": "system", "content": prompt}
    messages = [system_message]

    response = Client(host=host).chat(model=model, messages=messages, options= {"seed": 42, "top_p": 0.9, "temperature": 0 })

    return response, prompt, db_results

def get_questions_answers_contexts(store_name:str="documents"):
    """ Read the list of questions and answers and return a ragas dataset for evaluation """
    # URL of the file
    url = 'https://raw.githubusercontent.com/cgrodrigues/rag-intro/main/coldf_question_and_answer.psv'

    # Fetch the file from the URL
    response = requests.get(url)
    data = response.text

    # The encoders
    context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

    # Get the Vector Database Client
    _, vector_store = init_chroma_db(store_name=store_name)

    # Split the data into lines
    lines = data.split('\n')

    # Split each line by the pipe symbol and create tuples
    rag_dataset = []

    for line in lines[1:]:
        if line.strip():  # Ensure the line is not empty
            question, reference_answer = line.split('|')
            response, _, db_results = get_inference(question, context_encoder, context_tokenizer, vector_store)

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