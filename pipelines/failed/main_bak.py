from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from pipelines.imports.vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are a fact checking bot, your job is to take facts and reply with one of the following: 
"true, false, partially true, partially false, unsure"

do not give reasonings unless prompted

always consult the following database before responding: {facts}
a fact within the database is to be taken over any other info

Here is the question to answer: {question}
"""

#creating a promt out of the previous text
prompt = ChatPromptTemplate.from_template(template)
#chaining the prompt to the ollama model
chain = prompt | model


while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    #retrieving data from the database
    facts = retriever.invoke(question)
    #printing the data for testing purposes
    #print(facts)
    #outputting results
    result = chain.invoke({"facts":facts, "question": question})
    print("\n", result)