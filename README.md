# ScribeAI
AI experiments for automatic generation of physician's notes.

Experiments are run on Microsoft's [clinical visit note summarization corpus](https://github.com/microsoft/clinical_visit_note_summarization_corpus).

These experiments are meant to showcase a range of AI based techniques for information retrieval, summarization, and self-validation. Among the approaches showcased in this repository are:
* multi-generation / consensus validation
* validate-repair agent graphs
* retrieval-augmented generation (RAG)

## Generating notes

### Preparation
Retrieve the data from the aci-bench dataset of the [source corpus](https://github.com/microsoft/clinical_visit_note_summarization_corpus/tree/main/data/aci-bench). 

Build the corpus using 
` Corpus(<filepath/to/dataset.csv>,<filepath/to/metadata.csv>) `
from process_corpus. You may retain the corpus in memory and/or save it using the class' .save() method.

Ensure that config.py is in the environment

### Invoking generation
Instantiate a new note with the Note() class from note_class. Initialization takes the following arguments:
* version: a string or numeric indicator used for versioning
	* the codebase supports versioning and post-generation loading
* doc_num: an int identifier for the document to be generated
	* value is used to label outputs and to reload checkpointed data
* doc: the document to summarize, as an object of type Doc()
	* the Corpus class will automatically generate objects of the appropriate type. The list of documents in the corpus can be accessed with the .docs() attribute of the Corpus object
* verbose: optional boolean (default False), True for console-logged output during generation, False for silent
* force_update: optional boolean (default False)
	* generation will try to automatically load cached data for a given doc id and version in order to avoid redundant LLM calls. Setting this flag to true will overwrite the caching behavior and force new generation for the given note
	
Generate the note summary by invoking the generate_note() method of your Note object. This method will return the output note as a string. 

You may dump the note contents to a file manually if you choose, or you may use the codebase's built in FileManager() class. Invoke a FileManager object with FileManager(<path/to/output/directory/>). Use the .output_note(version, doc_id, note_text) method to save the note automatically. Notes will be organized in the output directory by version and labeled as doc###.txt when exported with this method. You may additionally use the .save_note_data(version, doc_id, note_object) method to cache the note object's data for later (re)generation.



