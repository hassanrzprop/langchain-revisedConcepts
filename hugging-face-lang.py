from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain, LLMChain
from dotenv import load_dotenv
import os
load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_TOKKEN")
)
# prompt="give the recipe of cake, it should be in steps"

# # ********Prompt template**********
# # 1) 
# langPromptTemplate=PromptTemplate.from_template("tell us the qualities of {textInput}")
# promptOuntput=langPromptTemplate.invoke({"textInput":"Dog"})

# # 2)method 2
# langprompt2=PromptTemplate("recipe of delicious {pizzaType}pizza ",input_variables=["pizzaType"])



# # chat model is used to 
# chat_model=ChatHuggingFace(llm=llm)
# res=chat_model.invoke(promptOuntput)
# print(res)



# chaining 
p1=PromptTemplate("translate the following text into French{InputText}",input_variables=["InputText"])
p2=PromptTemplate("What is feeling of this expression{InputText}",input_variables=["InputText"])
chain1=LLMChain(llm=llm,prompt=p1)
chain2=LLMChain(llm=llm,prompt=p2)
chain=SimpleSequentialChain(chains=[chain1,chain2])
result=chain.invoke("How are you?")
print(result)
