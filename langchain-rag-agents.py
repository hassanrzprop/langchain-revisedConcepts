from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
from langchain.tools import tool
import os
from dotenv import load_dotenv
load_dotenv()



llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                         google_api_key=os.getenv("GOOGLE_API_KEY"))

search = TavilySearchResults(tavily_api_key=os.getenv("TAVILY_API_KEY"))
# tool for search performing
loader = WebBaseLoader("https://faizanmotorsports.pk/")

# 4 steps of implementing RAG app
# 1) loading document
docs = loader.load()
# 2) Text spliting of loaded document
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
# 3) and 4) embedding and storing it into vector store

vector = FAISS.from_documents(documents, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

# part-2
# RAG based tool for gcu
retriever = vector.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "faizan_motosports_store_search",
    "Search for information about faizan motosports store. For any questions about faizan motorsports sotre, you must use this tool!",
)




tools = [search, retriever_tool]

prompt = hub.pull("hwchase17/openai-functions-agent")

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# it is used to handle history mangment of questions
message_history = ChatMessageHistory()


agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id:message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)
while True:
    agent_with_chat_history.invoke(
        {"input": input("How can I help you today? : ")},
        config={"configurable": {"session_id": "test123"}},
    )
