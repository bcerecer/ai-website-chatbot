from langchain_community.document_loaders import FireCrawlLoader  
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from openai import OpenAI
import os

def get_crawl_data(url):
    load_dotenv()

    loader = FireCrawlLoader(
        api_key=os.getenv('FIRECRAWL_API_KEY'), # Note: Replace 'YOUR_API_KEY' with your actual FireCrawl API key
        url=url,  # Target URL to crawl
        mode="crawl"  # Mode set to 'crawl' to crawl all accessible subpages
    )
    
    docs = loader.load()

    return docs

def setup_vector_store(docs):
    if not docs:
        raise ValueError("No documents were loaded. Please check the URL and the FireCrawl API key.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(documents=splits, embedding=OllamaEmbeddings())
    return vectorstore

def answer_user_prompt(question, vectorstore):
    load_dotenv()
    
    docs = vectorstore.similarity_search(query=question)
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    system_message = """You are a friendly assistant. Your job is to answer the user's question based on the documentation provided below."""
    
    user_message = f"Docs:\n\n{docs}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    )

    if response and response.choices:
        response_content = response.choices[0].message.content.strip()
        print(f"Answer: {response_content}\n")
    else:
        raise ValueError("The OpenAI API response did not contain response data.")

    return 

if __name__ == "__main__":
    url = 'https://firecrawl.dev'
        
    try:
        # Scrape data
        docs = get_crawl_data(url)
        
        # Setup vector store
        vectorstore = setup_vector_store(docs)
        
        while True:
            # Prompt the user for a question
            question = input("\nDo you have any questions about the website?\n")
            
            if question.lower() in ["no", "n"]:
                print("Thank you! Have a great day!")
                break
            
            # Answer the user's question
            answer_user_prompt(question, vectorstore)
    except Exception as e:
        print(f"An error occurred: {e}")
