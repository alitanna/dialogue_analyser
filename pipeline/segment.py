import re
from config import MAX_WORDS_PER_CHUNK

def clean_text(text: str) -> str:
    """Removes filler words while preserving punctuation."""
    # Regex to catch common fillers like 'uh', 'um', etc.
    text = re.sub(r'\b(uh|um|hmm|ah|uhm)\b', '', text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()

def group_and_chunk_segments(diarization_result):
    """
    Groups segments by speaker to transform text into structured analytical information[cite: 56].
    """
    # CRITICAL: Diarization returns a dict, we need the 'segments' list
    segments = diarization_result.get("segments", [])
    if not segments:
        return []

    speaker_turns = []
    current_turn = None

    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        
        if current_turn is None:
            current_turn = {"speaker": speaker, "parts": [text], "count": len(text.split())}
            continue

        # Merge if same speaker and below word limit to avoid information loss [cite: 187, 259]
        if speaker == current_turn["speaker"] and current_turn["count"] < MAX_WORDS_PER_CHUNK:
            current_turn["parts"].append(text)
            current_turn["count"] += len(text.split())
        else:
            speaker_turns.append({
                "speaker": current_turn["speaker"], 
                "text": " ".join(current_turn["parts"])
            })
            current_turn = {"speaker": speaker, "parts": [text], "count": len(text.split())}

    # Add final turn
    if current_turn:
        speaker_turns.append({
            "speaker": current_turn["speaker"], 
            "text": " ".join(current_turn["parts"])
        })

    return speaker_turns