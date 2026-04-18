import os
import pickle
from dataclasses import dataclass

from langchain.agents.structured_output import ResponseFormat
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from process_corpus import Corpus, Doc
from file_manager import *
from system_prompts import *

from pydantic import BaseModel, Field
from typing import List, Literal
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ChiefComplaint_ResponseFormat:
    chief_complaint: str

@dataclass
class ChiefComplaintArb_ResponseFormat:
    reasoning: str
    option_number: int

@dataclass
class DemographicsContext:
    patient_forename: str or None
    patient_surname: str or None
    patient_age: str or None
    patient_gender: str or None

class HPICitationSchema(BaseModel):
    statement: str = Field(description="Statement regarding history of patient illness")
    justification: List[str] = Field(description="Transcript context supporting statement regarding patient history")

class HPIOutputSchema(BaseModel):
    items: List[HPICitationSchema] = Field(
        description="List of statement-justification pairs"
    )

class ROSOutputSchema(BaseModel):
    conclusion: Literal["endorses","denies","undetermined"]

default_model = init_chat_model(config.model,temperature=0.2)

hpi_agent = create_agent(
    model=default_model,
    system_prompt=sp_hpi,
    context_schema=DemographicsContext,
    tools=[HPIOutputSchema],
    response_format=HPIOutputSchema
)

chief_complaint_agent = create_agent(
    model=default_model,
    system_prompt=sp_chief_complaint,
    #temperature=0.2,
    response_format = ChiefComplaint_ResponseFormat
)

chief_complaint_arbitrator = create_agent(
    model=default_model,
    system_prompt=sp_arb_chief_complaint,
    #temperature=0.2,
    response_format = ChiefComplaintArb_ResponseFormat
)

llm = llm = init_chat_model(
    model=config.model,
    temperature=0.2
)

ros_validation_prompt = ChatPromptTemplate.from_messages([
    ("system", sp_ros_validation),
    ("user", "{inputs}")
])

ros_validation_model = llm.with_structured_output(ROSOutputSchema)
ros_validation_agent = ros_validation_prompt | ros_validation_model

if __name__ == '__main__':

    version = "v0.1"
    doc_num = 0

    fm = FileManager("./ai_responses")

    with open("ms_cnvsc_acibench_train.corpus", "rb") as f:
        corpus = pickle.load(f)

    invocation = {
        "messages": [{"role": "user", "content": corpus.docs[1].dialogue}]
    }

    if not fm.response_exists(version,doc_num):
        response = chief_complaint_agent.invoke(invocation)
        fm.save_response(version,doc_num,response)
    else:
        try:
            response = fm.load_response(version,doc_num)
            print("WARNING: response exists. Raising cached response. Update version number to regenerate response.")
        except Exception as e:
            response = chief_complaint_agent.invoke(invocation)
            fm.save_response(version, doc_num, response)

    print(corpus.docs[doc_num].note)
    print(response["structured_response"].chief_complaint)

