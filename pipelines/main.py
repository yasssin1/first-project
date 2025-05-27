#the kkmimport
from langchain_ollama.llms import OllamaLLM
#for prompt generating
from langchain_core.prompts import ChatPromptTemplate
#info retriever we made in vector.py
from pipelines.imports.vector import retriever
from typing import List, Union, Generator, Iterator

#class definition
class Pipeline:
    #initialisation, values to be filled later
    def __init__(self):
        self.basic_rag_pipeline = None
        self.chain = None

    async def on_startup(self):
        #on start script for setting up constants and values

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
        self.chain = prompt | model
        

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    #defining the pipe here
    #model_id, messages and body go unused, maybe remove later
    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]: #return type, maybe should be string only?
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        #debugging
        print(messages)
        print(user_message)

        question = user_message
        facts = retriever.invoke(question)
        response = self.chain.invoke({"facts":facts, "question": question})

        return response