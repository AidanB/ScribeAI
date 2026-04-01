import pandas
import pickle
import re
import os

from langchain_core.vectorstores import InMemoryVectorStore
from nltk.tokenize import word_tokenize
from collections import Counter

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

SKIP_TOKENS = set([",","."])
RE_SENT_SPLIT = re.compile(r"[\.\!\?]")

class Doc():
    def __init__(self, corpus_row):
        self.dataset = corpus_row["dataset"]
        self.encounter_id = corpus_row["encounter_id"]
        self.dialogue = corpus_row["dialogue"]
        self.note = corpus_row["note"]

        self.build_ngrams()

        self.calculate_embeddings(patient_only=True)

    # not called on init because of data availability timing, must be called after Doc init to access doc.encounter_id
    def get_metadata(self,metadata_row):
        self.patient_forename = metadata_row["patient_firstname"].item()
        self.patient_surname = metadata_row["patient_familyname"].item()
        self.patient_gender = metadata_row["patient_gender"].item()
        self.patient_age = metadata_row["patient_age"].item()
        self.doctor_name = metadata_row["doctor_name"].item()

    def calculate_embeddings(self,patient_only=False):
        dialogue_turns = []
        doctor_turns = []
        patient_turns = []
        for line in self.dialogue.splitlines():
            try:
                speaker,turn = line.split("] ", maxsplit=1)
            except Exception:
                speaker = "unknown"
                turn = line

            dialogue_turns.append(line.strip())

            speaker = speaker.strip("[")
            match speaker:
                case "doctor":
                    doctor_turns.append(line)
                case "patient":
                    patient_turns.append(line)
                case _:
                    pass

        turns = []
        embeddings = []
        vector_store = InMemoryVectorStore(embeddings_model)
        if patient_only:
            for turn in patient_turns:
                turns.append(turn)
                embeddings.append(embeddings_model.embed_query(turn))
                #vector_store.add_texts(patient_turns)
        else:
            for turn in dialogue_turns:
                turns.append(turn)
                embeddings.append(embeddings_model.embed_query(turn))
                #vector_store.add_texts(dialogue_turns)

        self.turns = turns
        self.embeddings = embeddings

    def build_ngrams(self):
        for line in self.dialogue.splitlines():
            try:
                line.split("] ", maxsplit=1)[1]
            except Exception:
                pass #print(line)

        all_sents = []

        for line in self.dialogue.splitlines():
            try:
                turn = line.split("] ", maxsplit=1)[1]
            except IndexError: # this try/catch is necessary for exactly one broken document in the corpus
                turn = line

            sents = RE_SENT_SPLIT.split(turn)
            for sent in sents:
                clean_sent = [x for x in sent.strip().split(" ") if x not in SKIP_TOKENS and len(x)]
                all_sents.append(clean_sent)



        self.unigrams = Counter([" ".join(y) for x in all_sents for y in x])
        self.bigrams = Counter([" ".join(y) for x in all_sents for y in zip(x,x[1:])])
        self.trigrams = Counter([" ".join(y) for x in all_sents for y in zip(x,x[1:],x[2:])])


class Corpus():
    def __init__(self, corpus_df, metadata_df):
        self.corpus_df = self.read_df(corpus_df)
        self.metadata_df = self.read_df(metadata_df)
        self.docs = []
        self.unigrams = Counter()
        self.bigrams = Counter()
        self.trigrams = Counter()

        for row in self.corpus_df.iterrows():
            doc = Doc(row[1])
            metadata_row = self.metadata_df.loc[self.metadata_df['encounter_id'] == doc.encounter_id]
            doc.get_metadata(metadata_row)
            self.docs.append(doc)
            self.unigrams.update(doc.unigrams)
            self.bigrams.update(doc.bigrams)
            self.trigrams.update(doc.trigrams)

    def read_df(self,df_filepath):
        return pandas.read_csv(df_filepath,dtype=str) # setting dtype to string to standardize patient_age and empty cell values

    def save(self,filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(self,filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    data_subdirec = "microsoft-clinical_visit_note_summarization_corpus/aci-bench/challenge_data/"

    #training_corpus = Corpus(os.path.join(data_subdirec,"train.csv"),os.path.join(data_subdirec,"train_metadata.csv"))
    #test_corpus = Corpus(os.path.join(data_subdirec,"clinicalnlp_taskB_test1.csv"),os.path.join(data_subdirec,"clinicalnlp_taskB_test1_metadata.csv"))
    mini_test_corpus = Corpus(os.path.join(data_subdirec,"clinicalnlp_taskB_test1_mini.csv"),os.path.join(data_subdirec,"clinicalnlp_taskB_test1_metadata.csv"))

    #training_corpus.save("ms_cnvsc_acibench_train.corpus")
    mini_test_corpus.save("ms_cnvsc_acibench_test_mini.corpus")
