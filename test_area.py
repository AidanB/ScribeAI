from process_corpus import *
from agent_definitions import *
from file_manager import *
from system_prompts import *
from note_class import *
from utils import *

if __name__ == '__main__':
    version = "v1"

    fm = FileManager("./test-set_responses/")

    corpus = Corpus.load("ms_cnvsc_acibench_test_mini.corpus")

    # Single doc iteration
    i = 0
    curr_doc = corpus.docs[0]
    existing_note_data = fm.load_note_data(version, i)
    curr_note = Note.load_from_dict(existing_note_data, version=version, doc_num=i, doc=curr_doc, verbose=True)

    curr_note.get_hpi()
    fm.save_note_data(version,i,curr_note)


    """
    # Full iteration
    for i,doc in enumerate(corpus.docs):
        existing_note_data = fm.load_note_data(version,i)
        curr_note = Note.load_from_dict(existing_note_data,version=version,doc_num=i,doc=doc,verbose=True) #if existing_note_data else Note(version,i,doc,verbose=True)

        curr_note.get_chief_complaint()
        fm.save_note_data(version,i,curr_note)
    """