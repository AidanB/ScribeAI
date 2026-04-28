# ScribeAI
AI experiments for automatic generation of physician's notes.

All code and architecture human designed and written, except where otherwise noted by code comments. As this project is meant as a demonstration of ability, no LLMs were used for the coding of any primary functions and no AI was consulted for architectural or project design.

Experiments are run on Microsoft's [clinical visit note summarization corpus](https://github.com/microsoft/clinical_visit_note_summarization_corpus).

These experiments are meant to showcase a range of AI based techniques for information retrieval, summarization, and self-validation. Among the approaches showcased in this repository are:
* multi-generation / consensus validation
* validate-repair agent graphs
* retrieval-augmented generation (RAG)

## Generating notes

### Preparation
Retrieve the data from the aci-bench dataset of the [source corpus](https://github.com/microsoft/clinical_visit_note_summarization_corpus/tree/main/data/aci-bench). Alternatively, any dataset may be used so long as it matches the data structure of this corpus.

Corpus data is stored in a custom class. Objects of this class can be automatically built and saved using the interactive interface, see below.

### Invoking generation
Invocation of the generation pipeline is handled by an interactive interface, available through generate.py. This interface will walk you through building a new Corpus object for your data corpus, if you have not already built one. 

Once a Corpus has been built for your desired dataset, you may use the same interface to run the note generation process. You may run the entire corpus, or a subset of documents. 

The results of a run will be saved automatically in the directory configured under config.py. 



