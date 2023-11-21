import os
import json
from langchain.llms import SagemakerEndpoint
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS

class Content_Handler():
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        input_str = json.dumps({"inputs": prompt,
                                "parameters": {**model_kwargs}})
        return input_str.encode('utf-8')

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generated_text"]


def chat_bot():
    try:
        # check whether index has been already created
        if not os.path.isdir("pdf-index"):
            print("Index some data to get started")
            print("""
            You can add the pdfs you want to index in docs folder,
            then execute the connection.py file to index the pdf into vector database
            """)
            return

        # set the length of history
        max_history_len = 10
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local("pdf-index", embeddings=embeddings)
        content_handler = Content_Handler()
        llm = SagemakerEndpoint(
            endpoint_name=os.environ["ENDPOINT_NAME"],
            region_name=os.environ["REGION_NAME"],
            model_kwargs={"max_new_tokens": 700, "top_p": 0.9, "temperature": 0.6},
            content_handler=content_handler,
            credentials_profile_name="llm"
        )
        history = []
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type='stuff',
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True
        )
        while 1:
            print("Enter Your query")
            query = input()
            resp = chain({"question": query, "chat_history": history})
            history.append((query, resp["answer"]))
            while len(history) > max_history_len:
                history.pop(0)

            print("Answer")
            print(resp["answer"])

            print("Source documents")
            print(resp['source_documents'], "\n\n")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    print("LLMs on Custom Data")
    chat_bot()
