import gc
import torch
from transformers import pipeline
from config import SENTIMENT_MODEL, DEVICE

def analyze_sentiment(speaker_turns):
    """
    Advanced sentiment analysis using Probabilistic Neutrality and GPU Batching.
    Follows Dissertation Section 3.2.4 [cite: 356-363].
    """
    print(f"Loading {SENTIMENT_MODEL} with Batch Optimization...")
    
    # Using a larger batch_size (e.g., 16) significantly speeds up GPU inference
    classifier = pipeline(
        "sentiment-analysis", 
        model=SENTIMENT_MODEL, 
        device=0 if DEVICE=="cuda" else -1,
        batch_size=16 
    )

    # 1. Filter segments that need AI analysis
    # We still keep the '?' rule because questions are functionally neutral in dialogue
    texts_to_analyze = []
    indices_to_analyze = []

    for i, turn in enumerate(speaker_turns):
        text = turn["text"]
        
        # Enhanced Neutral Override Rule [cite: 361-362]
        # If it's a question or very short, we mark it NEUTRAL immediately to save GPU time
        if '?' in text or len(text.split()) < 4:
            turn["sentiment"] = "NEUTRAL"
            turn["sentiment_score"] = 1.0
        else:
            texts_to_analyze.append(text)
            indices_to_analyze.append(i)

    # 2. Batch Inference (The speed booster)
    if texts_to_analyze:
        print(f"Analyzing {len(texts_to_analyze)} segments in parallel...")
        results = classifier(texts_to_analyze, truncation=True)

        for idx, res in zip(indices_to_analyze, results):
            label = res["label"].upper()
            score = res["score"]

            # --- THE "SOFT-TRINARY" UPGRADE ---
            # If the model is not confident (e.g., < 85%), it's likely a 
            # neutral observation rather than a strong emotion.
            if score < 0.85:
                speaker_turns[idx]["sentiment"] = "NEUTRAL"
            else:
                speaker_turns[idx]["sentiment"] = label
            
            speaker_turns[idx]["sentiment_score"] = round(score, 4)

    print("Freeing VRAM for Summarization...")
    del classifier
    gc.collect()
    torch.cuda.empty_cache()
    
    return speaker_turns