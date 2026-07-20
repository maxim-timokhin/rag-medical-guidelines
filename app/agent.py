import os
from dotenv import load_dotenv

from google.adk.workflow import Workflow, node
from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.apps import App
from google.genai import types
from google import genai

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Initialize embeddings and database using the absolute path relative to the workspace root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_dir = os.path.join(BASE_DIR, "medquad_chroma")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=openai_api_key
)
db = Chroma(
    persist_directory=db_dir,
    embedding_function=embeddings
)

client = genai.Client(api_key=gemini_api_key)

def get_query_text(node_input) -> str:
    if isinstance(node_input, str):
        return node_input
    if hasattr(node_input, "parts") and node_input.parts:
        return "".join(part.text for part in node_input.parts if part.text)
    if isinstance(node_input, dict) and "parts" in node_input:
        parts = node_input["parts"]
        return "".join(part.get("text", "") for part in parts if isinstance(part, dict) and "text" in part)
    return str(node_input)

@node
def retrieve_docs(ctx: Context, node_input: types.Content):
    query = get_query_text(node_input)
    if not query:
        return Event(output="", state={"context": ""})
    
    # Retrieve documents from Chroma
    docs = db.similarity_search(query, k=5)
    
    formatted_docs = []
    for idx, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        content = doc.page_content
        formatted_docs.append(f"--- Document {idx+1} (Source: {source}) ---\n{content}")
    
    context_str = "\n\n".join(formatted_docs)
    
    # Pass the query as the output to the next node and store context in state
    return Event(output=query, state={"context": context_str})

@node
def generate_answer(ctx: Context, node_input: str):
    query = node_input
    context = ctx.state.get("context", "")
    
    if not context or not query:
        answer = "there is no source."
        yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=answer)]))
        yield Event(output=answer)
        return
        
    prompt = f"""You are a strict medical assistant.
You must answer the user's question using ONLY the provided Context.
If the Context does not contain the answer or is not relevant to the question, you must reply with exactly "there is no source." and nothing else.
Do not use any external knowledge.
Provide a citation to the source folder name (e.g. 3_GHR_QA) for the information you used.

Context:
{context}

Question: {query}
Answer:"""

    try:
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0)
        )
        answer = response.text.strip()
    except Exception as e:
        answer = f"Error generating content: {e}"
        
    yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=answer)]))
    yield Event(output=answer)

root_agent = Workflow(
    name="root_agent",
    edges=[
        ('START', retrieve_docs),
        (retrieve_docs, generate_answer)
    ],
    description="Medical assistant RAG application based on MedQuAD.",
)

app = App(
    root_agent=root_agent,
    name="app",
)
