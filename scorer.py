# scorer.py
from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def evaluate_coherence(original, improved):
    emb1 = embedder.encode(original, convert_to_tensor=True)
    emb2 = embedder.encode(improved, convert_to_tensor=True)
    score = util.pytorch_cos_sim(emb1, emb2).item()
    return round(score * 100, 2)

def feedback_score(text):
    # Heuristic scoring based on length, clarity etc.
    score_dict = {
        "Tone": "Professional",
        "Clarity": "Clear" if len(text.split()) < 250 else "Verbose",
        "Grammar": "Good" if len(text.split('.')) > 5 else "Needs Work",
        "Purpose": "Strong" if "research" in text.lower() or "goal" in text.lower() else "Unclear"
    }
    return score_dict

