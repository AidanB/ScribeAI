import config
from process_corpus import *
from agent_definitions import *
from file_manager import *
from system_prompts import *
from note_class import *
from utils import *

if __name__ == '__main__':
    version = "v2"

    fm = FileManager(config.target_path)

    corpus = Corpus.load("ms_cnvsc_acibench_test_mini.corpus")

    """
    # Single doc iteration
    i = 3
    curr_doc = corpus.docs[i]
    existing_note_data = fm.load_note_data(version, i)
    curr_note = Note.load_from_dict(existing_note_data, version=version, doc_num=i, doc=curr_doc, verbose=True, force_update=True)

    #curr_note.get_ros()
    #print(curr_note.format_note())
    curr_note.get_hpi()
    fm.save_note_data(version,i,curr_note)
    #output_note = curr_note.generate_note()
    #fm.save_note_data(version,i,curr_note)
    #fm.output_note(version,i,output_note)
    """

    # Full iteration, force update
    for i,doc in enumerate(corpus.docs):
        if i<4:
            continue

        curr_note = Note(version=version,doc_num=i,doc=doc,verbose=True,force_update=True)

        output_note = curr_note.generate_note()
        fm.save_note_data(version,i,curr_note)
        fm.output_note(version,i,output_note)
