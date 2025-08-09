# app.py
import streamlit as st
import document_processor_openai as dp_openai 
import json
import traceback
import pandas as pd
import time

# --- Helper Functions ---
def count_short_nodes(node, threshold, count=0):
    if len(node.get('text', '')) < threshold:
        count += 1
    for child in node.get('children', []):
        count = count_short_nodes(child, threshold, count)
    return count

# --- Page Config ---
st.set_page_config(page_title="Legal Document Parser", page_icon="🏛️", layout="wide")

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = []

# --- UI Layout ---
st.title("🏛️ Thai Legal Document Parser (v6.0 - Parser-First)")
st.markdown("Analyzes the hierarchical structure of Thai legal documents using a robust **Parser-First** approach.")

# --- Model Configuration ---
st.sidebar.header("⚙️ Analysis Configuration")
model_name = dp_openai.MODEL_NAME
st.sidebar.info(f"**LLM:** `{model_name}`")
st.sidebar.success("""
**New Architecture:**
1.  **Parser:** Python code first finds all potential headers with 100% accurate locations.
2.  **LLM:** OpenAI then organizes this clean list into a hierarchical tree.
""")
# [!!! REMOVED !!!] - Prompt editor is no longer needed.

# --- File Uploader and Analysis Execution ---
uploaded_file = st.file_uploader("Upload a Thai legal document (.txt)", type=['txt'])

if uploaded_file is not None:
    document_text = uploaded_file.getvalue().decode('utf-8')
    char_count = len(document_text)
    st.info(f"📁 **{uploaded_file.name}** | Total characters: **{char_count:,}**")

    if st.button(f"Run Analysis", type="primary"):
        st.session_state.analysis_result = None
        st.session_state.debug_info.clear()
        
        intermediate_results_container = st.empty() # Placeholder for potential future use

        with st.status(f"Running Parser-First analysis with {model_name}...", expanded=True) as status:
            try:
                api_key = st.secrets["OPENAI_API_KEY"]
                final_result = dp_openai.run_openai_pipeline(
                    document_text=document_text, 
                    api_key=api_key, 
                    status_container=status,
                    debug_info=st.session_state.debug_info
                )
                
                st.session_state.analysis_result = {
                    "final": final_result,
                    "debug": st.session_state.debug_info,
                    "file_name": uploaded_file.name
                }
                status.update(label="✅ Analysis complete!", state="complete", expanded=False)
                st.success("🎉 Successfully generated the hierarchical structure!")
                time.sleep(1)
                st.rerun()

            except Exception as e:
                st.session_state.analysis_result = None
                status.update(label="Fatal Error", state="error", expanded=True)
                st.error(f"An unexpected error occurred during processing: {e}")
                st.code(traceback.format_exc())

# --- Display Results ---
if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    final_result_data = result["final"]
    debug_info = result["debug"]
    file_name = result["file_name"]

    st.header("📄 Analysis Results")

    if "error" in final_result_data or not final_result_data.get("tree"):
        st.error(f"Analysis failed: {final_result_data.get('error', 'Unknown error')}. Please check the debug log for more details.")
    else:
        short_node_threshold = 15
        total_short_nodes = sum(count_short_nodes(node, short_node_threshold) for node in final_result_data.get('tree', []))
        
        timings_list = [item for item in debug_info if "performance_timings" in item]
        if timings_list:
            timings = timings_list[0]['performance_timings']
            st.subheader("📊 Performance Summary")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Step 1 (Parser)", f"{timings.get('step1_parser_duration', 0):.2f} s")
            col2.metric("Step 2 (LLM)", f"{timings.get('step2_llm_duration', 0):.2f} s")
            col3.metric("Step 3 (Post-proc)", f"{timings.get('step3_postprocess_duration', 0):.2f} s")
            col4.metric(f"Short Nodes (<{short_node_threshold} chars)", f"{total_short_nodes}",
                        help=f"Number of nodes with less than {short_node_threshold} characters of text.",
                        delta=f"-{total_short_nodes}" if total_short_nodes > 0 else None,
                        delta_color="inverse" if total_short_nodes > 0 else "off")

    tab1, tab2 = st.tabs(["✔️ Final Tree (JSON)", "🐞 Detailed Debug Log"])

    with tab1:
        st.markdown("""<style>.stJson > div {max-height: 600px; overflow-y: auto;}</style>""", unsafe_allow_html=True)
        st.json(final_result_data, expanded=True)
        st.download_button(
           label="Download Tree (JSON)",
           data=json.dumps(final_result_data, indent=2, ensure_ascii=False),
           file_name=f"{file_name.split('.')[0]}_pipeline_tree.json",
           mime="application/json",
        )

    with tab2:
        st.info("This log contains Regex parser results and the raw LLM response.")
        st.json({"pipeline_logs": debug_info}, expanded=False)
        st.download_button(
           label="Download Debug Log (JSON)",
           data=json.dumps(debug_info, indent=2, ensure_ascii=False),
           file_name=f"{file_name.split('.')[0]}_pipeline_debug.json",
           mime="application/json",
        )