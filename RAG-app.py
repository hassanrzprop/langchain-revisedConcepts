from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI,GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

llm=GoogleGenerativeAI(model="gemini-1.5-flash",google_api_key=os.getenv("GOOGLE_API_KEY"))

# indexing consist of four steps :     
# 1)loading data
# 2)emmbeding data with llm's embedding model
# 3)splitting data 
# 4) storing data in memory or disk through vector stores

# 1)loading data
try:
    loader=TextLoader("data.txt");
except Exception as e:
    print("Error while loading file=",e);


# 2)emmbeding data
embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 3)splitting data
# use a samller chunk size to manage tokens limit
text_splitter=CharacterTextSplitter(chunk_size=500,chunk_overlap=100)


# 4)vector store  create the index with specified embedding model and text splitter
index_creater=VectorstoreIndexCreator(
    embedding=embedding,
    text_splitter=text_splitter
)
index= index_creater.from_loaders([loader])


# Now the retrival of  data from the index variable
response=index.query("does gcu have hostels?",llm=llm)
print(response)
