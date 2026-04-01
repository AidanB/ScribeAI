import re

RE_METADATA = re.compile(r"\(?(M|m)etadata[:;]?\)?")
target_path = "./test-set_responses/"
embeddings_path = "./embeddings/"
global_version = "v1"

similarity_threshold = 0.6

model = "gpt-5-nano"

symptoms_by_system = {
    "Constitutional":["chills","fatigue","fever","weight gain","weight loss"],
    "HEENT":["hearing loss","sinus pressure","vision changes"],
    "Respiratory":["cough","shortness of breath","wheezing"],
    "Cardiovascular": ["chest pain", "pain while walking (claudication)", "edema", "palpitations"],
    "Gastrointestinal": ["abdominal pain", "blood in stool", "constipation", "diarrhea", "heartburn", "loss of appetite", "nausea", "vomiting"],
    "Genitourinary": ["painful urination (dysuria)", "excessive amount of urine (polyuria)", "urinary frequency"],
    "Metabolic/Endocrine": ["cold intolerance", "heat intolerance", "excessive thirst (polydipsia)", "excessive hunger (polyphagia)"],
    "Neurological": ["dizziness", "extremity numbness", "extremity weakness", "headaches", "seizures", "tremors"],
    "Psychiatric": ["anxiety", "depression"],
    "Integumentary": ["breast discharge", "breast lump", "hives", "mole change(s)", "rash", "skin lesion"],
    "Musculoskeletal": ["back pain", "joint pain", "joint swelling", "neck pain"],
    "Hematologic": ["easily bleeds", "easily bruises", "lymphedema", "issues with blood clots"],
    "Immunologic": ["food allergies", "seasonal allergies"]
}

import math


def cosine_similarity(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of the same length")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a ** 2 for a in vec1))
    magnitude2 = math.sqrt(sum(b ** 2 for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0  # Return 0 for zero vectors to avoid division by zero

    return dot_product / (magnitude1 * magnitude2)