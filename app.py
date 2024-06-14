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

    response = client.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    )

    print(response.choices[0].message.content)

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
            question = input("Do you have any questions about the website? ")
            
            if question.lower() in ["no", "n"]:
                print("Thank you! Have a great day!")
                break
            
            # Answer the user's question
            answer_user_prompt(question, vectorstore)
            
            # Ask if the user has any other questions
            another_question = input("Do you have any other question? ")
            
            if another_question.lower() in ["no", "n", "No", "NO"]:
                print("Thank you! Have a great day!")
                break
        
    except Exception as e:
        print(f"An error occurred: {e}")
