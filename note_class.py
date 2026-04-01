import config
from collections import Counter
from dataclasses import asdict
from scipy.spatial.distance import cosine

from process_corpus import Corpus, Doc, embeddings_model
from file_manager import FileManager
from agent_definitions import *
from ros_logic import get_ros_candidates
from utils import *
from hpi_graph import generate_hpi


"""
Class used to store pertinent information and functions for generating clinical notes.

Class maintains its own versioning, which can be used for data recall to avoid redundant LLM calls.

version: version id, as a str
doc_num: index of document in corpus
doc: object of Doc class (see Corpus) which contains data from the original corpus document (transcript, metadata, etc.)
verbose: flags whether to log information to console
force_update: overwrite version control on True, force run of all LLM components

TODO: implement logic for remaining note sections (physical exam, results, assessment and plan)
"""

class Note():
    def __init__(self,version,doc_num,doc,verbose=False,force_update=False):
        self.version = version
        self.doc_num = doc_num
        self.force_update = force_update

        self.doc = doc

        self.log = print if verbose else lambda *a, **k: None # logging fxn

        self.log(f"### Doc {self.doc_num:03d} ###")
        self.log(f"### Version {self.version} ###")

        self.chief_complaint = None
        self.chief_complaint_candidates = [] # unsurfaced intermediate steps, useful for debugging
        self.chief_complaint_arb_reasoning = None

        # variables for clinical note components
        self.hpi = None
        self.ros = None
        self.physical_exam = None
        self.vitals_review = None
        self.vitals_results = None
        self.assessment_and_plan = None

    # serialize own attributes, used for saving/loading as some components are not pickle-able
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

    # control logic for data recall functionality
    # all LLM-based components check for existing data before re-run, unless force_update=True
    def handle_force_update(self,version,section):
        if self.__dict__[section]:
            if not self.force_update:
                print(f"Warning: data {section} already exists for doc {self.doc_num} of version {self.version}. Please pass force_update=True to force update.")
                print(f"Existing value for {section}: {self.__dict__[section]}\n")
                return True

        return False

    # plain text output of resulting clinical note
    def format_note(self):
        output_note = f"""CHIEF COMPLAINT
        
{self.chief_complaint.strip()}

HISTORY OF PRESENT ILLNESS

{self.hpi.strip()}

REVIEW OF SYSTEMS

{self.ros.strip()}

PHYSICAL EXAMINATION

VITALS REVIEWED

RESULTS

ASSESSMENT AND PLAN
        """

        return output_note

    # wrapper function to return metadata for use in some LLM calls
    # TODO this can probably be cleaned up with better prompt templating
    def get_metadata_context(self):
        metadata = {"patient_forename":self.doc.patient_forename,
                "patient_surname":self.doc.patient_surname,
                "patient_age":self.doc.patient_age,
                "patient_gender":self.doc.patient_gender}

        # TODO bandaid solution for blank metadata values, fix should be relocated to corpus builder
        for k,v in metadata.items():
            if type(v) != str:
                metadata[k] = ""

        return metadata

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

    # Generates the HISTORY OF PATIENT ILLNESS
    # Algorithm is generate-validate-repair loop, see hpi_graph for more details
    def get_hpi(self):
        if self.handle_force_update(self.version,"hpi"):
            return

        self.log("Processing information for HISTORY OF PATIENT ILLNESS")

        self.hpi = generate_hpi(self.doc.dialogue,self.get_metadata_context())
        self.hpi = config.RE_METADATA.sub("",self.hpi)

    # Handles generation of REVIEW OF SYSTEMS
    # Uses simple RAG system: dialogue turns from the patient transcript are embedded in a vector store
    # Each symptom queries the vector store. Candidate retrievals that exceed a certain threshold are evaluated by LLM
    def get_ros(self):
        if self.handle_force_update(self.version,"ros"):
            return

        self.log("Processing information for REVIEW OF SYSTEMS")

        candidates = get_ros_candidates(self.doc)

        conclusions = {}
        for symptom,justifications in candidates.items():
            inputs = f"Symptom: {symptom}\n"
            for justification in justifications:
                inputs += justification+"\n"

            results = ros_validation_agent.invoke({"inputs":inputs})
            conclusion = results.conclusion
            self.log(f"{symptom}: {conclusion}")
            conclusions[symptom] = conclusion

        ros = ""
        for system in config.symptoms_by_system:
            temp = []
            for symptom in config.symptoms_by_system[system]:
                if symptom in conclusions:
                    if conclusions[symptom] != "undetermined":
                        temp.append((symptom,conclusions[symptom]))
            if len(temp):
                temp_str = f"{system}: "
                for i,(s,c) in enumerate(temp):
                    if i == 0:
                        t = f"{c[0].upper()}{c[1:]} {s}, "
                    else:
                        t = f"{c} {s}, "
                    temp_str += t
                temp_str = temp_str[0:-2] + "."
                ros += temp_str+"\n"

        self.ros = ros.strip()
        self.log(self.ros)

    # call all generative components
    def generate_note(self):
        self.get_chief_complaint()
        self.get_hpi()
        self.get_ros()

        return self.format_note()