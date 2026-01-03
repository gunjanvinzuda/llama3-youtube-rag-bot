from dotenv import load_dotenv
import os

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["LANGCHAIN_API_KEY"]= os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv('LANGCHAIN_ENDPOINT')
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ["USER_AGENT"] = os.getenv('USER_AGENT', 'RAGYouTubeBot')
os.environ["LANGCHAIN_PROJECT"] = os.getenv('LANGCHAIN_PROJECT', 'YouTube-RAG-Bot')


def youtube_transcript_loader(video_id: str) -> str:
    try:
        youtube_transcript_api = YouTubeTranscriptApi()
        transcript_list = youtube_transcript_api.fetch(video_id)
        return " ".join(snippet.text for snippet in transcript_list)
    except TranscriptsDisabled:
        raise Exception("[Error] Caption not available in English.")
    except Exception as e:
        raise Exception(f"[Error] During fetching transcript {e}")


def create_indexing(transcript: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([transcript])
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_stores = FAISS.from_documents(docs, embeddings)
    return vector_stores

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def youtube_chat_bot():
    video_id = input("Enter a YouTube video id: ").strip()

    print("\n[Process]Loading video transcript and building knowledge base...")
    transcript = youtube_transcript_loader(video_id)
    vector_stores = create_indexing(transcript)
    retriever = vector_stores.as_retriever(search_kwargs={"k":4})
    template = """
    Act as a helpful chat assistant. 
    Answer given question based on ONLY provided context. 
    If context is not sufficient say "I don't Know"
    Context: {context}
    Questions: {question}    
    """
    llm = ChatOllama(model="llama3", temperature=0.2)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    rag_chain = (
        {
            "context": retriever | format_docs, "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n[Process]Video processed. Now you can ask queries.")
    print("\nType quit or q to stop.")
    while True:
        query = input("\nUser: ").strip()

        if query.lower() in ['quit', 'q']:
            print("Goodbye!")
            break

        if not query:
            continue

        try:
            response = rag_chain.invoke(query)
            print(f"\nBot: {response}")
        except Exception as e:
            raise Exception(f"[Error] During retrival: {e}")


if __name__ == "__main__":
    print("---- YouTube Video Chatbot ----")
    youtube_chat_bot()