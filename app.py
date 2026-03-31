import streamlit as st
import pandas as pd
import plotly.express as px
import torch
from pipeline.run import run_full_analysis

st.set_page_config(page_title="Dialogue Analyzer - Conversation Analysis System", page_icon="🎙️", layout="wide")

# --- 1. CLEAN CALLBACK (NO VRAM) ---
def update_ui_callback(message, percent):
    status_text.text(f"Current Stage: {message}")
    main_progress_bar.progress(percent / 100.0)

# --- 2. MAIN UI ---
st.title("🎙️ Dialogue Analyzer : Conversation Analysis System")
st.markdown("---")

st.header("Input Section")
url_input = st.text_input("YouTube URL:", placeholder="Enter link...")
analyze_btn = st.button("🚀 Analyze Video")

# Store results
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

# --- 3. EXECUTION LOGIC ---
if analyze_btn and url_input:
    st.session_state.analysis_result = None
    
    main_progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        with st.spinner("Executing modular pipeline..."):
            result = run_full_analysis(url_input, progress_callback=update_ui_callback)
            st.session_state.analysis_result = result
            st.success("Analysis Complete!")
    except Exception as e:
        st.error(f"Pipeline Error: {e}")
    finally:
        main_progress_bar.empty()
        status_text.empty()

# --- 4. RESULTS DISPLAY ---
if st.session_state.analysis_result:
    res = st.session_state.analysis_result
    
    tab1, tab2, tab3 = st.tabs(["📄 Transcript", "📝 Summaries", "📊 Visualization"])
    
    # --- Transcript ---
    with tab1:
        st.subheader("Speaker-Separated Transcript")
        for turn in res["transcript"]:
            sent = turn.get("sentiment", "NEUTRAL")
            score = turn.get("sentiment_score", 0.0)
            color = "green" if sent == "POSITIVE" else "red" if sent == "NEGATIVE" else "gray"
            st.markdown(f"**{turn['speaker']}** [:{color}[{sent} ({score:.2f})]]: {turn['text']}")

    # --- Summaries ---
    with tab2:
        st.subheader("In-Depth Conversation Summaries")
        st.info(res["summary"])
        
        st.markdown("---")
        for spk, smry in res.get("speaker_summaries", {}).items():
            with st.expander(f"Detailed Insights from {spk}"):
                st.write(smry)

    # --- Visualization ---
    with tab3:
        st.subheader("Semantic Insights & Sentiment Analysis")
        df = pd.DataFrame(res["transcript"])
        
        if not df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                chart_data = df.groupby(["speaker", "sentiment"]).size().unstack(fill_value=0)
                st.bar_chart(chart_data)

            with col2:
                sentiment_map = {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}
                df['polarity'] = df['sentiment'].map(sentiment_map)
                df['turn_number'] = range(1, len(df) + 1)
                
                fig = px.line(
                    df,
                    x="turn_number",
                    y="polarity",
                    color="speaker",
                    title="Emotional/Semantic Trajectory",
                    line_shape="hv",
                    markers=True
                )
                fig.update_yaxes(
                    tickvals=[-1, 0, 1],
                    ticktext=["Negative", "Neutral", "Positive"]
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Insufficient data for visualization.")