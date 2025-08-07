# app.py

import streamlit as st
import document_processor
import json

st.set_page_config(page_title="Thai Legal Document Processor", page_icon="⚖️", layout="wide")
st.title("⚖️ Thai Legal Document RAG Pre-processor")
st.write("Upload a Thai legal text file to analyze its structure and generate summaries for a Hybrid RAG system.")

# Use session_state to store results across reruns
if 'final_result' not in st.session_state:
    st.session_state.final_result = None

uploaded_file = st.file_uploader("Upload the .txt file", type=['txt'])

if uploaded_file is not None:
    if st.button("Begin Full Analysis"):
        st.session_state.final_result = None
        document_text = uploaded_file.getvalue().decode('utf-8')
        st.write(f"File Info: **{len(document_text):,}** characters")
        
        with st.status("Running full analysis pipeline...", expanded=True) as status:
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
                # Run the final, complete pipeline
                final_json_result = document_processor.run_full_pipeline(
                    document_text=document_text, 
                    api_key=api_key,
                    status_container=status
                )
                # Store the result in session state
                st.session_state.final_result = final_json_result
                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
                st.success("🎉 All tasks completed successfully!")
            except Exception as e:
                st.session_state.final_result = None
                status.update(label="A critical error occurred", state="error", expanded=True)
                st.error(f"An unexpected error occurred during the process: {e}")
                st.code(traceback.format_exc())

# Display results and download button if available in session state
if st.session_state.final_result:
    result = st.session_state.final_result
    
    st.subheader("📊 Analysis Result Preview")
    st.json(result, expanded=False) # Show a collapsed preview

    st.download_button(
       label="⬇️ Download Final JSON",
       data=json.dumps(result, indent=2, ensure_ascii=False),
       file_name=f"{uploaded_file.name.split('.')[0]}_final.json",
       mime="application/json",
    )