# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
# from langchain.schema import AIMessage,HumanMessage,SystemMessage
# from dotenv import load_dotenv
# import os

# llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash",google_api_key=os.getenv("GOOGLE_API_KEY"))

# prompt_template=ChatPromptTemplate.from_messages(
#     [
#         SystemMessage(content="This chat is about pshcological human satisfaction facts"),
#         SystemMessage(content="you should be strictly replied only to relevant questions don't reply to other irrelevant questions  and say sorry and your answers should be in calm manner which provide user friendly interaction with my app"),   
#     ]
# )

# while True:
#     user_input=input("YOU: ")
#     if user_input=="exit":
#         break;
#     print(user_input)
#     prompt_template.append(HumanMessage(content=user_input))
#     prompt=prompt_template.format()
#     print("PROMPT:",prompt)
#     response=llm.invoke(prompt)
#     print("LLM RESPONSE:",response)
#     prompt_template.append(AIMessage(content=response.content))
# # format helps in understanding the whole context in one variable

# # managing tokens for input is compulsary as our chat history increases we should do something to control tokens

# #  THIS WHOLE PROCESS WAS MANUAL NOW USING LANGHCAIN MEMMORY
from langchain.memory import ConversationBufferMemory,ConversationBufferWindowMemory,ConversationSummaryMemory,ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()


llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash",google_api_key=os.getenv("GOOGLE_API_KEY"))

# # managing tokens for input is compulsary as our chat history increases we should do something to control tokens

memory=ConversationBufferWindowMemory(k=2)

#  ConversationSummaryMemory used to make summarized questions form to reduce tokens and cover long range of questions but accuracy reduced
memory1=ConversationSummaryMemory(llm=llm)

# here we can define to summarize whole questions in given token limit
memory2=ConversationSummaryBufferMemory(llm=llm, max_token_limit=1000)

chain=ConversationChain(llm=llm,memory=memory1)

while True:
    user_input=input("YOU: ")
    if user_input=="exit":
        break;
    response=chain.invoke(user_input)
    print("LLM RESPONSE:",response)
