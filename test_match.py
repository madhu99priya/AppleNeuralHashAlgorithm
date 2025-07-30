from src.matcher import match_face

image_path = "images/incoming/unknown_001.jpg"
db_path = "embeddings/criminals.json"

match_id, score = match_face(image_path, db_path)

if match_id:
    print(f"\n✅ Match Found: {match_id} (Score: {score:.3f})")
else:
    print(f"\n❌ No match found. Best score: {score:.3f}")
