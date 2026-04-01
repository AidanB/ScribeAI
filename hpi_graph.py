from typing import Annotated, Literal, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cosine
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

import config
from process_corpus import *
from system_prompts import *

load_dotenv()

sim_threshold = config.similarity_threshold

verbose = True
logger = print if verbose else lambda *a, **k: None
debug_state = 1

coverage_iter = 0
repair_iter = 0

llm = init_chat_model(
    model="gpt-5-nano",
    temperature=0.2
)

class HPICitationSchema(BaseModel):
    statement: str
    justifications: List[str] = Field(min_length=1)

class ValidationStatus(BaseModel):
    statement_id: int
    status: Literal["valid", "invalid", "revised"] = "valid"
    errors: List[Literal["No valid citation","Insufficient semantic match"]] = []

class InfoCluster(BaseModel):
    spans: List[str]
    summary: str

class InfoClusterOutput(BaseModel):
    clusters: List[InfoCluster]

class StatementOutput(BaseModel):
    statements: List[HPICitationSchema]

class State(BaseModel):
    source_transcript: str

    patient_forename: str | None
    patient_surname: str | None
    patient_age: str | None
    patient_gender: str | None

    info_clusters: List[InfoCluster] = Field(default_factory=list)
    cluster_coverage: List[bool] = Field(default_factory=list)
    cluster_summ_embeddings: List[List[float]] = Field(default_factory=list)

    statements: List[HPICitationSchema] = Field(default_factory=list)
    statement_embeddings: List[List[float] | str] = Field(default_factory=list)

    validations: List[ValidationStatus] = Field(default_factory=list)

    iteration: int = 0

cluster_model = llm.with_structured_output(InfoClusterOutput)
statement_model = llm.with_structured_output(StatementOutput)
repair_model = llm.with_structured_output(StatementOutput)

cluster_prompt = ChatPromptTemplate.from_messages([
    ("system", sp_hpi_cluster),
    ("user", "{transcript}")
])

statement_prompt = ChatPromptTemplate.from_messages([
    ("system", sp_hpi_statement),
    ("user", "{transcript}"),
    ("user", "{patient_forename}"),
    ("user", "{patient_surname}"),
    ("user", "{patient_age}"),
    ("user", "{patient_gender}"),
    (MessagesPlaceholder(variable_name="repair_text",optional=True))
])

repair_prompt = ChatPromptTemplate.from_messages([
    ("system", sp_hpi_repair),
    ("user", "{transcript}"),
    ("user", "{patient_forename}"),
    ("user", "{patient_surname}"),
    ("user", "{patient_age}"),
    ("user", "{patient_gender}"),
    ("user","{curr_summary}")
])

cluster_agent = cluster_prompt | cluster_model
statement_agent = statement_prompt | statement_model
repair_agent = repair_prompt | repair_model

def debug_invoke(agent, payload, name):
    print(f"\n### {name} INPUT ###\n", payload)
    result = agent.invoke(payload)
    print(f"\n### {name} OUTPUT ###\n", result)
    return result

def build_clusters(state: State) -> State:
    if debug_state == 1:
        return state

    text = state.source_transcript

    result = debug_invoke(cluster_agent,{
        "transcript": state.source_transcript
    }, "cluster_agent")

    return state.model_copy(update={
        "info_clusters": result.clusters
    })

def embed_clusters(state: State) -> State:
    if debug_state == 1:
        return state

    summ_embeddings = []
    for cluster in state.info_clusters:
        summ_embeddings.append(embeddings_model.embed_query(cluster.summary))

    return state.model_copy(update={"cluster_summ_embeddings": summ_embeddings})

def generate_statements(state: State) -> State:
    text = state.source_transcript

    result = debug_invoke(statement_agent, {
        "transcript": state.source_transcript,
        "patient_forename": state.patient_forename,
        "patient_surname": state.patient_surname,
        "patient_gender": state.patient_gender,
        "patient_age": state.patient_age,
    }, "statement_agent")

    return state.model_copy(update={"statements": result.statements})

def embed_statements(state: State) -> State:
    embeddings = []
    for citation in state.statements:
        statement = citation.statement
        justifications = citation.justifications
        if "metadata" in justifications:
            embeddings.append("metadata") # don't bother generating embeddings for demographics info, won't validate properly anyway
        else:
            embeddings.append(embeddings_model.embed_query(statement))

    return state.model_copy(update={"statement_embeddings": embeddings})

def format_repair(state: State) -> State:
    output_summ = ""

    invalid_ids = [x.statement_id for x in state.validations if x.status == "invalid"]
    for i,statement in enumerate(state.statements):
        if i in invalid_ids:
            output_summ += f"<{statement.statement}>["
            for just in statement.justifications:
                output_summ += f"{just},"
            output_summ = output_summ[:-1] # remove final comma
            output_summ += "]\n\n"
        else:
            output_summ += f"{statement.statement}\n\n"

    return output_summ



def validate_coverage(state: State) -> State:
    if debug_state == 1:
        return state

    cluster_covered = []
    for i,summ_embedding in enumerate(state.cluster_summ_embeddings):
        logger(f"Validating coverage: {state.info_clusters[i].summary}",end=" ")
        covered = False
        for statement_embedding in state.statement_embeddings:
            if statement_embedding == "metadata":
                continue
            sim = 1 - cosine(summ_embedding,statement_embedding)
            logger(sim,end=" ")
            if sim > sim_threshold:
                covered = True
                continue
        cluster_covered.append(covered)

    return state.model_copy(update={"cluster_coverage":cluster_covered})

def validate_justification(state: State) -> State:
    statement_justified = []

    for i in range(len(state.statement_embeddings)):
        statement_txt = state.statements[i].statement
        logger(f"Validating statement: {statement_txt}")

        justifications = [x for x in state.statements[i].justifications if len(x)>20]

        statement_e = state.statement_embeddings[i]
        if statement_e == "metadata":
            statement_justified.append(True)
            logger("Validated (metadata)")
            continue

        just_e = [embeddings_model.embed_query(justification) for justification in state.statements[i].justifications]

        curr_best = 0
        for j_e in just_e:
            sim = cosine(statement_e,j_e)
            logger(sim,end=" ")
            if sim > curr_best:
                curr_best = sim

        validated = True if curr_best > sim_threshold else False
        logger(validated)
        statement_justified.append(validated)

    validations = []
    for i,v in enumerate(statement_justified):
        status = "valid" if v else "invalid"

        validation = ValidationStatus(**{"statement_id":i,
            "status":status,
            "errors":["Insufficient semantic match"] if not v else []
        })
        validations.append(validation)

    logger(validations)
    return state.model_copy(update={"validations": validations})

def repair_coverage(state: State) -> State:
    if debug_state == 1:
        return state

    message = ""
    for section in state.statements:
        message += section.statement + " " + section.justifications

    message += """The following conditions of the patient were found to have insufficient coverage in the summary as written. Please revise your summary to include or expand the coverage of these conditions. Please return a revised version of the full summary.\n"""

    for i,x in enumerate(state.cluster_coverage):
        if x == False:
            message += f"{state.info_clusters[i].summary}\n"

    debug_invoke(statement_agent,
                 {"transcript":state.source_transcript,"repair_text":message},
                 "statement_agent COVERAGE REPAIR")
    pass

def repair_statements(state: State) -> State:
    curr_summ = format_repair(state)
    logger("Repairing: ")
    logger(curr_summ)

    result = debug_invoke(repair_agent,
                          {"transcript":state.source_transcript,
                           "patient_forename": state.patient_forename,
                           "patient_surname": state.patient_surname,
                           "patient_gender": state.patient_gender,
                           "patient_age": state.patient_age,
                           "curr_summary":curr_summ},
                        "repair_agent"
    )

    pointer = 0

    new_statements = []
    for validation in state.validations:
        id = validation.statement_id
        if validation.status == "valid":
            new_statements.append(state.statements[id])
        elif validation.status == "invalid":
            if result.statements[pointer].statement != "<remove>":
                new_statements.append(result.statements[pointer])
            pointer += 1

    return state.model_copy(update={"statements":new_statements,"validations":[],"iteration":state.iteration+1})




def coverage_logic(state: State) -> State:
    if False in state.cluster_coverage:
        return "repair"
    else:
        return "continue"

def exit_logic(state: State) -> str:
    if all(v.status == "valid" for v in state.validations):
        return "end"

    if state.iteration >= 2:
        return "end"

    return "repair"

def format_output(state: State) -> str:
    output = ""
    for statement in state.statements:
        output += f"{statement.statement}\n\n"

    return output.strip()

builder = StateGraph(State)

builder.add_node("build_clusters", build_clusters)
builder.add_node("embed_clusters", embed_clusters)
builder.add_node("generate", generate_statements)
builder.add_node("embed_statements", embed_statements)
builder.add_node("validate_coverage", validate_coverage)
builder.add_node("validate_statements", validate_justification)
builder.add_node("repair_coverage", repair_coverage)
builder.add_node("repair_statements", repair_statements)

builder.set_entry_point("build_clusters")

builder.add_edge("build_clusters", "embed_clusters")
builder.add_edge("embed_clusters","generate")
builder.add_edge("generate", "embed_statements")
builder.add_edge("embed_statements", "validate_coverage")
builder.add_conditional_edges("validate_coverage",
                              coverage_logic,
                              {
                                  "repair":"repair_coverage",
                                  "continue":"validate_statements"
                               })
builder.add_edge("repair_coverage","embed_statements")
builder.add_conditional_edges("validate_statements",
    exit_logic,
    {
        "repair": "repair_statements",
        "end": END
    }
)
builder.add_edge("repair_statements","embed_statements")

hpi_graph = builder.compile()

print(hpi_graph.get_graph().draw_ascii())

def generate_hpi(transcript,metadata):
    logger(metadata)
    logger(metadata["patient_forename"])

    result = hpi_graph.invoke({
        "source_transcript": transcript,
        "patient_forename": metadata["patient_forename"],
        "patient_surname": metadata["patient_surname"],
        "patient_age": metadata["patient_age"],
        "patient_gender": metadata["patient_gender"],
    })

    final_state = State(**result)

    return format_output(final_state)


if __name__ == '__main__':
    corpus = Corpus.load("ms_cnvsc_acibench_test_mini.corpus")

    curr_doc = corpus.docs[1].dialogue

    result = hpi_graph.invoke({
        "source_transcript": curr_doc
    })

    final_state = State(**result)

    print(format_output(final_state))
