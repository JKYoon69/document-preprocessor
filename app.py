# app.py

import streamlit as st
import document_processor
import json
import traceback

st.set_page_config(page_title="RAG Pre-processor", page_icon="⚖️", layout="wide")
st.title("⚖️ Thai Legal Document RAG Pre-processor")
st.write("This tool analyzes a Thai legal document, builds a hierarchical structure, and generates summaries for each section, creating a final JSON output for a RAG system.")

if 'final_result' not in st.session_state:
    st.session_state.final_result = None

uploaded_file = st.file_uploader("Upload a Thai legal .txt file", type=['txt'])

if uploaded_file is not None:
    if st.button("Begin Full Analysis"):
        st.session_state.final_result = None
        document_text = uploaded_file.getvalue().decode('utf-8')
        
        with st.status("Running full analysis pipeline...", expanded=True) as status:
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
                # Pass the uploaded_file object to the pipeline
                final_json_result = document_processor.run_final_pipeline(
                    document_text=document_text, 
                    api_key=api_key,
                    status_container=status,
                    uploaded_file=uploaded_file 
                )
                st.session_state.final_result = {
                    "data": final_json_result,
                    "file_name": uploaded_file.name
                }
                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
                st.success("🎉 All tasks completed successfully!")

            except Exception as e:
                st.session_state.final_result = None
                status.update(label="A critical error occurred", state="error", expanded=True)
                st.error(f"An unexpected error occurred during the process: {e}")
                st.code(traceback.format_exc())

# Display results and download button if available in session state
if st.session_state.final_result:
    result = st.session_state.final_result["data"]
    file_name = st.session_state.final_result["file_name"]
    
    st.subheader("📊 Final Processed Result Preview")
    st.json(result, expanded=False)

    st.download_button(
       label="⬇️ Download Final JSON",
       data=json.dumps(result, indent=2, ensure_ascii=False),
       file_name=f"{file_name.split('.')[0]}_final_processed.json",
       mime="application/json",
    )