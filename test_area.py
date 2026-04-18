import config
from process_corpus import *
from agent_definitions import *
from file_manager import *
from system_prompts import *
from note_class import *
from utils import *

if __name__ == '__main__':
    #run_type = "partial-corpus"
    run_type = "full-corpus"
    version = f"{config.global_version}_{run_type}_{config.model}"

    fm = FileManager(config.target_path)

    if run_type=="partial-corpus":
        corpus = Corpus.load("ms_cnvsc_acibench_test_mini.corpus")
    if run_type=="full-corpus":
        corpus = Corpus.load("ms_cnvsc_acibench_test.corpus")


    # Full iteration, force update
    for i,doc in enumerate(corpus.docs):
        curr_note = Note(version=version,doc_num=i,doc=doc,verbose=True,force_update=True)

        output_note = curr_note.generate_note()
        fm.save_note_data(version,i,curr_note)
        fm.output_note(version,i,output_note)
