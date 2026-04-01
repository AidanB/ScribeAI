import inspect
from collections import Counter
from dataclasses import asdict
from scipy.spatial.distance import cosine

from process_corpus import Corpus, Doc, embeddings_model
from file_manager import FileManager
from agent_definitions import *
from utils import *
from hpi_graph import generate_hpi


class Note():
    def __init__(self,version,doc_num,doc,verbose=False,force_update=False):
        self.version = version
        self.doc_num = doc_num
        self.force_update = force_update

        self.min_sim = .6 # hard coded minimum cosine similarity for now

        self.doc = doc

        self.log = print if verbose else lambda *a, **k: None

        self.log(f"### Doc {self.doc_num:03d} ###")
        self.log(f"### Version {self.version} ###")

        self.chief_complaint = None
        self.chief_complaint_candidates = []
        self.chief_complaint_arb_reasoning = None

        self.hpi = None
        self.systems_review = None
        self.physical_exam = None
        self.vitals_review = None
        self.vitals_results = None
        self.assessment_and_plan = None

    def serialize_data(self):
        serialized = self.__dict__
        serialized.pop("doc")
        serialized.pop("log")
        return serialized

    # This method was AI-generated. Didn't want to spend the time on fiddling with the minutiae of it and knew ChatGPT could do a perfectly fine job of it
    @classmethod
    def load_from_dict(cls,d,**kwargs):
        # handle case of None passed for existing data
        d = d or {}

        # Merge in correct precedence:
        # kwargs > saved dict > defaults
        merged = {**d, **kwargs}

        # Only pass valid __init__ args
        import inspect
        sig = inspect.signature(cls.__init__)
        valid_keys = set(sig.parameters) - {"self"}

        init_args = {
            k: v for k, v in merged.items()
            if k in valid_keys
        }

        # Construct object normally
        obj = cls(**init_args)

        # Handle any extra attributes not in __init__
        for k, v in merged.items():
            if k not in valid_keys:
                setattr(obj, k, v)

        return obj

    def handle_force_update(self,version,section):
        if self.__dict__[section]:
            if not self.force_update:
                print(f"Warning: data {section} already exists for doc {self.doc_num} of version {self.version}. Please pass force_update=True to force update.")
                print(f"Existing value for {section}: {self.__dict__[section]}\n")
                return True

        return False

    def format_note(self):
        output_note = f"""CHIEF COMPLAINT
        
        {self.chief_complaint.strip()}
        
        HISTORY OF PRESENT ILLNESS
        
        {self.hpi.strip()}
        
        REVIEW OF SYSTEMS
        
        PHYSICAL EXAMINATION
        
        VITALS REVIEWED
        
        RESULTS
        
        ASSESSMENT AND PLAN
        """

        return output_note

    def get_metadata_context(self):
        return {"patient_forename":self.doc.patient_forename,
                "patient_surname":self.doc.patient_surname,
                "patient_age":self.doc.patient_age,
                "patient_gender":self.doc.patient_gender}

    def get_chief_complaint(self):
        """
        Methodology: five generations, majority rules
        Discard generations that do not meet minimum requirements (capital first letter, final period)
        If no majority, fallback to LLMaaJ
        """
        if self.handle_force_update(self.version,"chief_complaint"):
            return

        generations = []
        counter = 0
        self.log("Generating candidates for CHIEF COMPLAINT...")
        while counter < 10:
            if not len(generations) == 5:
                invocation = {
                    "messages": [{"role": "user", "content": self.doc.dialogue}]
                }
                generation = chief_complaint_agent.invoke(invocation)["structured_response"].chief_complaint
                if (is_capital(generation[0])) and generation.endswith("."):
                    generations.append(generation)
                    self.log(f"CANDIDATE: {generation}")
                else:
                    self.log(f"REJECTED: {generation}")
            else:
                break

        self.chief_complaint_candidates = generations

        counts = Counter(generations)
        if counts.most_common(1)[0][1] >= 3:
            self.chief_complaint = counts.most_common(1)[0][0]
            self.chief_complaint_arb_reasoning = "N/A"
            self.log(f"Selected by majority rule: {self.chief_complaint}")
        else: # no majority consensus
            self.log("No majority consensus. Falling back to arbitrator agent.")
            for_arb = as_numbered_list(generations)

            invocation = {
                "messages": [{"role": "user", "content": self.doc.dialogue},
                             {"role": "user", "content": for_arb}]
            }

            response = chief_complaint_arbitrator.invoke(invocation)
            self.log(f"Model reasoning: {response["structured_response"].reasoning}")
            self.chief_complaint_arb_reasoning = response["structured_response"].reasoning
            selected_i = int(response["structured_response"].option_number) - 1 # revert to 0-indexing
            self.chief_complaint = generations[selected_i]
            self.log(f"CHIEF COMPLAINT: {self.chief_complaint}")

    def get_hpi(self):
        if self.handle_force_update(self.version,"hpi"):
            return

        self.log("Processing information for HISTORY OF PATIENT ILLNESS")

        self.hpi = generate_hpi(self.doc.dialogue,self.get_metadata_context())

    def get_ros(self):
        if self.handle_force_update(self.version,"ros"):
            return

        self.log("Processing information for REVIEW OF SYSTEMS")



    def generate_note(self):
        self.get_chief_complaint()
        self.get_hpi()

        return self.format_note()