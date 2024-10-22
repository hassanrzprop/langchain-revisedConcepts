from langchain.agents import initialize_agent,AgentType
from langchain_google_genai import GoogleGenerativeAI
from langchain.tools import tool
import os 
from dotenv import load_dotenv
load_dotenv()
  

# intialize LLM
llm=GoogleGenerativeAI(model="gemini-1.5-flash",google_api_key=os.getenv("GOOGLE_API_KEY"))

@tool
def add_numbers_tool(input_data:str)-> str:    # tool logic is written here
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

agent=initialize_agent(
    tools=[add_numbers_tool],     # tools list
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # choosing agents
    llm=llm,
    verbose=False,    # vervose option is for what is happening in the background to  see it.
    max_iterations=1   # repition of loop
)
agent.run("2,3")