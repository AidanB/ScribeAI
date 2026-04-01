import config
from process_corpus import embeddings_model
from ros_logic import symptoms_by_system
from file_manager import FileManager

"""
Pre-compute embeddings for the symptoms we will query the REVIEW OF SYSTEMS symptoms on.
Since queries are static across runs, no need to recompute them every run.
"""

embedded_queries = {}
version = "v1"
fm = FileManager(config.target_path)

for system in symptoms_by_system:
    for symptom in symptoms_by_system[system]:
        embedding = embeddings_model.embed_query(symptom)
        embedded_queries[symptom] = embedding

fm.save_query_embeddings(version,"ros",embedded_queries)