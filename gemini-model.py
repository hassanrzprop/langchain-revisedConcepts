#  we can get gemini key from google ai studio
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

llm=GoogleGenerativeAI(model="gemini-1.5-flash",google_api_key=os.getenv("GOOGLE_API_KEY"))


prompt=PromptTemplate(template="Create a story about two friends meet at tech seminar. Use these name as character{character}. your response should start with name after name place colon e.g name:",input_variables="[character]")
chain=prompt | llm 
response= chain.invoke({"character":" Ali and Hassan"})
print(response)