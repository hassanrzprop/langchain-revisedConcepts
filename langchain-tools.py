from  langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
load_dotenv()
import os
llm=GoogleGenerativeAI(model="gemini-1.5-flash",google_api_key=os.getenv("GOOGLE_API_KEY"))

prompt_template=PromptTemplate(
    input_variables=["input"],
    template="you are a tool caller you have to call the tool named add_numbers_tool in case if there any addition required,please don't send any    explanation while calling the function. just send the numbers  what user provided   e.g 2,5.even though user gave the sentence you have two find two numbers and pass to the function user input is:{input}\n"
)

@tool
def add_numbers_tool(input_data:str)-> str:
    """ addition of two numbers."""
    # print("add_numbers_tool input_data ",input_data)
    # return " your result is 10"
    try:
        numbers=input_data.split(",")
    except Exception as e:
        return input_data
    num1,num2=int(numbers[0]),int(numbers[1])
    result=num1 + num2
    return f" the sum of {num1} and {num2} is {result}"

chain=RunnableSequence(
    prompt_template,
    llm,
    add_numbers_tool
)
res=chain.invoke("i travel twenty kms towards south and 30kms towards north what will be my result")
print(res)