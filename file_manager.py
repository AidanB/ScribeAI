import json
import os
import pickle
import re
from glob import glob
from collections import defaultdict

import config

"""
Class handles file save/load operations for various other classes (notably Note), in addition to final output of generated clinical notes.

__init__: parent directory for outputs
"""

class FileManager:
    def __init__(self,target_directory):
        self.RE_DOC_ID = re.compile(r"doc(\d\d\d)")

        self.target_directory = target_directory
        self.responses = defaultdict(set)
        self.file_structure = defaultdict(set)

        if not os.path.isdir(self.target_directory):
            os.makedirs(self.target_directory)

        self.evaluate_dir()

    def evaluate_dir(self):
        subdirs = [x for x in os.scandir(self.target_directory) if x.is_dir()]

        for subdir in subdirs:
            files = glob(os.path.join(subdir.path,"doc*.response"))
            if len(files):
                for file in files:
                    doc_name = os.path.basename(file)
                    doc_id = int(self.RE_DOC_ID.findall(doc_name)[0][1])
                    self.responses[subdir.name].add(doc_id)
                    self.file_structure[subdir.name].add(doc_name)

    def response_exists(self,version,doc_id):
        self.evaluate_dir() # ensure directory info is up-to-date

        if version in self.responses and doc_id in self.responses[version]:
            return True
        else:
            return False

    def _generate_filepath(self,version,doc_id,filetype):
        filetype = "."+filetype if not filetype.startswith(".") else filetype
        filetype = filetype.lower()

        return os.path.join(self.target_directory,f"{version}",f"doc{doc_id:03d}{filetype}")

    def save_note_data(self,version,doc_id,note_obj):
        filepath = self._generate_filepath(version,doc_id,".json")

        serialized_data = note_obj.serialize_data()

        if not os.path.isdir(os.path.join(self.target_directory,version)):
            os.makedirs(os.path.join(self.target_directory,version))
        with open(filepath,"w+",encoding="utf8") as f:
            json.dump(serialized_data,f,ensure_ascii=False)

    def load_note_data(self,version,doc_id):
        try:
            filepath = self._generate_filepath(version,doc_id,".json")
            with open(filepath,"r",encoding="utf8") as f:
                note = json.load(f)

            return note
        except Exception as e:
            print(f"Warning: encountered error {e} when attempting to open file {doc_id}")
            return None

    def output_note(self,version,doc_id,note_text):
        filepath = self._generate_filepath(version,doc_id,".txt")

        with open(filepath,"w+",encoding="utf8") as f:
            f.write(note_text)

    def save_query_embeddings(self,version,query_type,embeddings_obj):
        filepath = os.path.join(config.embeddings_path,version)
        os.makedirs(filepath,exist_ok=True)

        filename = f"{query_type}.embeddings"
        filepath = os.path.join(filepath,filename)

        with open(filepath,"wb") as f:
            pickle.dump(embeddings_obj,f)

    def load_query_embeddngs(self,version,query_type):
        filepath = os.path.join(config.embeddings_path,version)
        filename = f"{query_type}.embeddings"
        filepath = os.path.join(filepath,filename)

        with open(filepath,"rb") as f:
            return pickle.load(f)

    def save_response(self,version,doc_id,response):
        filepath = self._generate_filepath(version,doc_id,".note")
        if not os.path.isdir(os.path.join(self.target_directory,version)):
            os.makedirs(os.path.join(self.target_directory,version))
        with open(filepath,"wb") as f:
            #print(response,file=f)
            pickle.dump(response,f)

    def load_response(self,version,doc_id):
        filepath = self._generate_filepath(version,doc_id,".note")
        with open(filepath,"rb") as f:
            response = pickle.load(f)

        return response

if __name__ == '__main__':
    fm = FileManager("./test/")
    print(fm.response_exists("v1",0))
    fm.save_response("v1",0,"This is a triumph.")
    print(fm.response_exists("v1",0))
    print(fm.response_exists("v1",1))
    print(fm.response_exists("v2",1))
    print(fm.responses)






