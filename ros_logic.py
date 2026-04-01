from langchain.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore

import config
from file_manager import FileManager
from process_corpus import Corpus, Doc

version = "v1"

symptoms_by_system = {
    "constitutional":["chills","fatigue","fever","weight gain","weight loss"],
    "heent":["hearing loss","sinus pressure","vision changes"],
    "respiratory":["cough","shortness of breath","wheezing"],
    "cardiovascular": ["chest pain", "pain while walking (claudication)", "edema", "palpitations"],
    "gastrointestinal": ["abdominal pain", "blood in stool", "constipation", "diarrhea", "heartburn", "loss of appetite", "nausea", "vomiting"],
    "genitourinary": ["painful urination (dysuria)", "excessive amount of urine (polyuria)", "urinary frequency"],
    "metabolic/endocrine": ["cold intolerance", "heat intolerance", "excessive thirst (polydipsia)", "excessive hunger (polyphagia)"],
    "neurological": ["dizziness", "extremity numbness", "extremity weakness", "headaches", "seizures", "tremors"],
    "psychiatric": ["anxiety", "depression"],
    "integumentary": ["breast discharge", "breast lump", "hives", "mole change(s)", "rash", "skin lesion"],
    "musculoskeletal": ["back pain", "joint pain", "joint swelling", "neck pain"],
    "hematologic": ["easily bleeds", "easily bruises", "lymphedema", "issues with blood clots"],
    "immunologic": ["food allergies", "seasonal allergies"]
}

fm = FileManager(config.target_path)
query_embeddings = fm.load_query_embeddngs(version,"ros")

def retrieve_turns(vector_store,query):
    return vector_store.similarity_search_with_score(query,k=5)


class VectorStoreWrapper:
    def __init__(self,turns,embeddings):
        self.turns = turns
        self.embeddings = embeddings
        self.embedded_queries = {k:v for k,v in zip(turns,embeddings)}

    def embed_documents(self,texts):
        return self.embeddings

    def embed_query(self,query_str):
        return query_embeddings[query_str]



if __name__ == '__main__':
    corpus = Corpus.load("ms_cnvsc_acibench_test_mini.corpus")

    curr_doc = corpus.docs[0]
    vs_passthrough = VectorStoreWrapper(curr_doc.turns,curr_doc.embeddings)

    vector_store = InMemoryVectorStore.from_texts(embedding=vs_passthrough,texts=curr_doc.turns)

    for query in query_embeddings:
        results = retrieve_turns(vector_store,query)

        # Filter by similarity score: .3 selected based on manual review of a few cases
        # Also filter for only turns >3 words. Anything shorter will never have enough context on its own for meaningful diagnosis

        results = [x[0].page_content for x in results if x[1]>.3 and len(x[0].page_content.split())>3] # .3 selected as similarity threshold based on a few empirical examples
        print(query)
        [print(x) for x in results]
        print()
