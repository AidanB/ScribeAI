import config
from process_corpus import *
from agent_definitions import *
from file_manager import *
from system_prompts import *
from note_class import *
from utils import *
from argparse import ArgumentParser
from glob import glob
import plyer

if __name__ == '__main__':
    #parser = ArgumentParser()
    #parser.add_argument()

    corpus_files = glob(os.path.join("./","*.corpus"),recursive=True)

    def print_corpus_files(corpus_files):
        print("Found the following corpus files in the working directory:")
        for i,filename in enumerate(corpus_files):
            print(f"{i}: {filename}")
        print("Type the number for the corpus file you wish to load. Or type 'new' to create a new corpus, or 'settings' to adjust settings.")

    def new_corpus_userloop():
        dummy_in = input("First, please select the .csv file for your dataset. Press Enter to continue, or type 'cancel' to return to the previous menu.")
        if dummy_in.strip().lower() == "cancel":
            return

        dataset_filepath = plyer.filechooser.open_file(multiple=False, filters=["*.csv"])
        dataset_filepath = dataset_filepath[0]

        if not dataset_filepath.endswith(".csv"):
            print("The file selected is not a .csv file. Corpora can only be loaded from .csv files.")
            return -1

        dummy_in = input("Next, please select the .csv file for the metadata file for this dataset.")
        if dummy_in.strip().lower() == "cancel":
            return

        metadata_filepath = plyer.filechooser.open_file(multiple=False, filters=["*.csv"])
        metadata_filepath = metadata_filepath[0]

        if not dataset_filepath.endswith(".csv"):
            print("The file selected is not a .csv file. Corpora can only be loaded from .csv files.")
            return -1

        print("Attempting to build corpus file.")
        print(f"Dataset: {dataset_filepath}")
        print(f"Metadata: {metadata_filepath}")

        try:
            corpus = Corpus(dataset_filepath,metadata_filepath)
            input("Corpus built successfully. Next, select a location to save the corpus. Press Enter to continue.")
            save_path = plyer.filechooser.save_file()[0]
            if not save_path.endswith(".corpus"):
                save_path = save_path + ".corpus"
            corpus.save(save_path)
            return corpus
        except Exception as e:
            print(f"Encountered the following exception when attempting to build the corpus: {e}")

        return

    if len(corpus_files):
        print_corpus_files(corpus_files)
        loaded = False
        while not loaded:
            user_input = input().strip()
            if user_input.isdigit():
                if int(user_input) <= len(corpus_files):
                    corpus_file = corpus_files[int(user_input)]
                    loaded = True
                else:
                    print("The number was not found within the list of available corpus files. Please select from one of the following options.")
                    print_corpus_files(corpus_files)
            elif user_input.lower() == "new":
                new_corpus_created = False
                while not new_corpus_created:
                    corpus_file = new_corpus_userloop()
                    if corpus_file == -1:
                        print("Let's try that again.")
                    elif corpus_file:
                        loaded = True
                        new_corpus_created = True
            elif user_input.lower() == "settings":
                print("Settings implementation pending development. Please alter configurations manually in config.py.")


    #run_type = "partial-corpus"
    run_type = "full-corpus"
    version = f"{config.global_version}_{run_type}_{config.model}"

    fm = FileManager(config.target_path)

    if type(corpus_file) == Corpus:
        corpus = corpus_file
    else:
        corpus = Corpus.load(corpus_file)

    print(f"Found {len(corpus.docs)} documents in the corpus.")

    def doc_range_userloop(corpus_size):
        determined = False

        while not determined:
            run_type = input("Select which documents to process, e.g. 0-4. Or hit Enter to process the whole corpus.\n")
            run_type = config.RE_NUMBER_RANGE.findall(run_type.strip())
            if len(run_type):
                run_type = run_type[0]
                a = int(run_type[0])
                try:
                    b = int(run_type[2])
                except ValueError:
                    b = 0

                if b and a > b:
                    print("The numbers provided are not in the correct order. Please input a range a-b where a is the first document to process and b is the last.")
                elif a >= corpus_size or b >= corpus_size:
                    print(f"The upper bound of the range exceeds the total size of the corpus. Please select an upper bound less than {corpus_size}")
                elif b:
                    to_run = (a,b)
                    determined = True
                else:
                    to_run = (a,a)
                    determined = True
            else:
                to_run = (0,corpus_size-1)

        print(f"Processing documents {to_run[0]} through {to_run[1]}.")
        return to_run

    to_run = doc_range_userloop(len(corpus.docs))

    # Full iteration, force update
    for i in range(to_run[0],to_run[1]+1):
        doc = corpus.docs[i]
        curr_note = Note(version=version,doc_num=i,doc=doc,verbose=True,force_update=True)

        output_note = curr_note.generate_note()
        fm.save_note_data(version,i,curr_note)
        fm.output_note(version,i,output_note)
