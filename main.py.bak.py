from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="llama3.2")

template = """
You are a fact checking bot, your job is to take facts and reply with one of the following: 
"true, false, partially true, partially false, unsure"

do not give reasonings unless prompted

always consult the following database before responding: "1+1=3"
a fact within the database is to be taken over any other info

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    result = chain.invoke({"question": question})
    print(result)