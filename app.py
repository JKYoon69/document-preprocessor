# app.py
import streamlit as st
import document_processor as dp_gemini # Original Gemini processor
import document_processor_openai as dp_openai # New OpenAI processor
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
st.title("🏛️ Thai Legal Document Parser (v5.1 - Verified Pipeline)")
st.markdown("Analyzes the hierarchical structure of Thai legal documents. The processing pipeline includes an algorithmic verification step to prevent LLM hallucinations.")

# --- Model Selection UI ---
st.sidebar.header("⚙️ Analysis Configuration")
selected_model = st.sidebar.radio(
    "Choose the LLM to use:",
    ("Gemini (gemini-1.5-flash)", "OpenAI (gpt-4.1-mini)"),
    key="model_selection"
)

# Display model-specific information
if "Gemini" in selected_model:
    model_name_display = "gemini-1.5-flash"
    st.sidebar.info("Uses Google's Gemini API. Optimized for speed and handling large contexts. Requires `GEMINI_API_KEY`.")
else:
    model_name_display = "gpt-4.1-mini"
    st.sidebar.info("Uses OpenAI's API. Potentially better for complex JSON formatting. Requires `OPENAI_API_KEY`.")

st.markdown(f"**Selected LLM:** `{model_name_display}`")


# --- Prompt Editor ---
with st.expander("📝 Edit Prompts for Each Step (Advanced)"):
    if 'prompt1' not in st.session_state:
        st.session_state.prompt1 = dp_gemini.PROMPT_ARCHITECT
    if 'prompt2' not in st.session_state:
        st.session_state.prompt2 = dp_gemini.PROMPT_SURVEYOR
    if 'prompt3' not in st.session_state:
        st.session_state.prompt3 = dp_gemini.PROMPT_DETAILER

    tab1, tab2, tab3 = st.tabs(["Step 1: Architect", "Step 2: Surveyor", "Step 3: Detailer"])
    with tab1:
        st.session_state.prompt1 = st.text_area("Architect Prompt", value=st.session_state.prompt1, height=250, key="p1")
    with tab2:
        st.session_state.prompt2 = st.text_area("Surveyor Prompt", value=st.session_state.prompt2, height=250, key="p2")
    with tab3:
        st.session_state.prompt3 = st.text_area("Detailer Prompt", value=st.session_state.prompt3, height=250, key="p3")


# --- File Uploader and Analysis Execution ---
uploaded_file = st.file_uploader("Upload a Thai legal document (.txt)", type=['txt'])

if uploaded_file is not None:
    document_text = uploaded_file.getvalue().decode('utf-8')
    char_count = len(document_text)
    st.info(f"📁 **{uploaded_file.name}** | Total characters: **{char_count:,}**")

    if st.button(f"Run Analysis with {model_name_display}", type="primary"):
        st.session_state.analysis_result = None
        st.session_state.debug_info.clear()

        intermediate_results_container = st.empty()
        def display_intermediate_result(result):
            with intermediate_results_container.container():
                st.write("---")
                llm_duration = next((item.get('llm_duration_seconds', 0) for item in st.session_state.debug_info if "step1_architect_response" in item), 0)
                st.write(f"✅ Step 1 (Architect) Complete! Found top-level structures. (LLM call: {llm_duration:.2f}s)")
                display_data = [{"type": n.get('type'), "title": n.get('title')} for n in result]
                st.dataframe(display_data)

        with st.status(f"Running 3-step analysis with {model_name_display}...", expanded=True) as status:
            try:
                # --- CONDITIONAL PIPELINE EXECUTION ---
                if "Gemini" in selected_model:
                    api_key = st.secrets["GEMINI_API_KEY"]
                    # ==========================================================
                    # 여기가 수정된 부분입니다
                    final_result = dp_gemini.run_gemini_pipeline(
                        document_text=document_text, api_key=api_key, status_container=status,
                        prompt_architect=st.session_state.prompt1, prompt_surveyor=st.session_state.prompt2,
                        prompt_detailer=st.session_state.prompt3, debug_info=st.session_state.debug_info,
                        intermediate_callback=display_intermediate_result
                    )
                    # ==========================================================
                else: # OpenAI
                    api_key = st.secrets["OPENAI_API_KEY"]
                    final_result = dp_openai.run_openai_pipeline(
                        document_text=document_text, api_key=api_key, status_container=status,
                        prompt_architect=st.session_state.prompt1, prompt_surveyor=st.session_state.prompt2,
                        prompt_detailer=st.session_state.prompt3, debug_info=st.session_state.debug_info,
                        intermediate_callback=display_intermediate_result
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
        st.error("Analysis failed. Please check the debug log for more details.")
    else:
        short_node_threshold = 15
        total_short_nodes = sum(count_short_nodes(node, short_node_threshold) for node in final_result_data.get('tree', []))

        timings_list = [item for item in debug_info if "performance_timings" in item]
        if timings_list:
            timings = timings_list[0]['performance_timings']
            st.subheader("📊 Performance Summary")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Step 1 (Architect)", f"{timings.get('step1_architect_duration', 0):.2f} s")
            col2.metric("Step 2 (Surveyor)", f"{timings.get('step2_surveyor_duration', 0):.2f} s")
            col3.metric("Step 3 (Detailer)", f"{timings.get('step3_detailer_duration', 0):.2f} s")
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
        st.info("This log contains raw LLM responses and performance data for each step.")
        st.json({"pipeline_logs": debug_info}, expanded=False)
        st.download_button(
           label="Download Debug Log (JSON)",
           data=json.dumps(debug_info, indent=2, ensure_ascii=False),
           file_name=f"{file_name.split('.')[0]}_pipeline_debug.json",
           mime="application/json",
        )