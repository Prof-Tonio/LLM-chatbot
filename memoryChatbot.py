import PyPDF2
from langchain.chains import LLMChain, ConversationChain
from langchain.llms import SagemakerEndpoint
from langchain.prompts import PromptTemplate
from langchain.llms.sagemaker_endpoint import LLMContentHandler
import os
import json
from langchain.memory import ConversationBufferWindowMemory
import requests
import connection


class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        input_str = json.dumps({"inputs": prompt, "parameters": {**model_kwargs}})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generated_text"]



template = """<s>[INST]You are an intelligent coding assitant, who specialses in helping with programming, documentation and answering technical concepts[/INST]</s>\n Please answer the following question: {question}"""
prompt = PromptTemplate.from_template(template)
print(prompt)
llm = SagemakerEndpoint(
    endpoint_name="lct0-shared-hugging-face-mistral-7b-instruct",
    region_name="ca-central-1",
    model_kwargs={"max_new_tokens": 500, "top_p": 0.8, "temperature": 0.01},
    # endpoint_kwargs={"CustomAttributes": 'accept_eula=true'},
    content_handler=ContentHandler(),
    credentials_profile_name="llm",
    verbose=True,
)

memory = ConversationBufferWindowMemory(k=99)
#text = connection.textHandler("/Users/272047234/Downloads/devops-de.pdf")
text = connection.urlHandler("https://rbcgithub.fg.rbc.com/pages/Innersource-Commons/helios-docs/docs/qtest/welcome")
memory.save_context({"input": f"Hi! Comprehend this content and the logic,{text}"},
                    {"output": "Hi I fully comprehend and ready to be tested on the content"})

memory.load_memory_variables({})
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=memory
)
continues = True
while continues:
    user_prompt = input("Enter a question!")
    if str(user_prompt) != "Stop":
        print(conversation.predict(input=user_prompt))
    else:
        continues = False
