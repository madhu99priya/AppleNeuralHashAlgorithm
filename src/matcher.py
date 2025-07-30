import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.embedder import get_embedding

def load_database(db_path):
    with open(db_path, "r") as f:
        data = json.load(f)
    return data

def match_face(image_path, db_path, threshold=0.65):
    """
    Compares input face to each person in DB and returns best match above threshold.
    """
    embedding = get_embedding(image_path)
    db = load_database(db_path)

    best_match = None
    best_score = -1

    for person in db:
        db_vector = np.array(person["embedding"])
        score = cosine_similarity([embedding], [db_vector])[0][0]

        if score > best_score:
            best_score = score
            best_match = person["person_id"]

    if best_score >= threshold:
        return best_match, best_score
    else:
        return None, best_score
