import os
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL

def generate_summaries(speaker_turns):
    """
    Uses the Groq API to generate highly structured, detailed summaries.
    Completely bypasses local VRAM limits and provides formatting.
    """
    print(f"--- Step 6: Generating Structured Summaries via Groq ({GROQ_MODEL}) ---")
    
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing from your .env file.")

    client = Groq(api_key=GROQ_API_KEY)

    # Group dialogue by speaker
    speaker_data = {}
    for turn in speaker_turns:
        spk = turn["speaker"]
        if spk not in speaker_data:
            speaker_data[spk] = []
        speaker_data[spk].append(turn["text"])

    per_speaker_summaries = {}
    all_speaker_texts = []

    # The System Prompt forces the model to act like an analytical note-taker
    system_prompt = (
        "You are an expert analytical assistant. Your job is to read the provided transcript "
        "and create a highly detailed, structured summary. Extract the core arguments, protocols, "
        "and actionable advice. Use bullet points and paragraphs for readability. "
        "Do not just compress the text; explain the concepts thoroughly based on the transcript."
    )

    # 1. Speaker-Level Summaries
    for speaker, text_list in speaker_data.items():
        print(f"Requesting detailed summary for {speaker}...")
        combined_text = " ".join(text_list)
        
        # Groq's 8192 token window safely handles ~6000 words in a single shot.
        # We slice at 5500 just to be absolutely safe.
        truncated_text = " ".join(combined_text.split()[:5500])
        
        if len(truncated_text.split()) < 30:
            per_speaker_summaries[speaker] = truncated_text
            continue

        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Please provide a detailed, structured summary of the points made by {speaker}:\n\n{truncated_text}"}
                ],
                temperature=0.3, # Lower temperature keeps the AI factual and focused
                max_tokens=1024  # Allows the AI to write a massive, detailed response
            )
            summary = response.choices[0].message.content
            per_speaker_summaries[speaker] = summary
            all_speaker_texts.append(f"{speaker}'s Summary:\n{summary}")
        except Exception as e:
            print(f"⚠️ Groq API Error for {speaker}: {e}")
            per_speaker_summaries[speaker] = "API Error: Could not generate summary."

    # 2. Final Master Summary
    print("Generating final master overview...")
    master_input = "\n\n".join(all_speaker_texts)
    
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are an executive summarizer. Combine the following speaker summaries into one master overview of the entire conversation. Highlight the main themes and conclusions using clear formatting and bullet points."},
                {"role": "user", "content": f"Create a master summary from these speaker notes:\n\n{master_input}"}
            ],
            temperature=0.3,
            max_tokens=1024
        )
        overall_summary = response.choices[0].message.content
    except Exception as e:
         print(f"⚠️ Groq API Error for Master Summary: {e}")
         overall_summary = "API Error: Could not generate master summary."

    return {"overall": overall_summary, "per_speaker": per_speaker_summaries}