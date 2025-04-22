import streamlit as st
from dotenv import load_dotenv # Import the function
load_dotenv() # Load variables from .env file into environment (if it exists)
import pandas as pd
from io import StringIO, BytesIO # Keep BytesIO for output
import time # For potential delays
import os # Often useful, good to have

# Import functions from local modules
from llm_integrations import get_references_from_llm
from utils import load_csv, prepare_output_csv

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Reference Extractor", layout="wide")
st.title("📚 Reference Extraction and Categorization")
st.markdown("""
Upload a CSV file containing text chunks (e.g., from Maulana Wahiduddin Khan's works).
The app will use the selected Large Language Model (LLM) to extract references based on a detailed schema.
""")

# --- Session State Initialization ---
# Use session state to keep track of results across reruns
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'results_data' not in st.session_state:
    st.session_state.results_data = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'current_llm_provider' not in st.session_state:
    st.session_state.current_llm_provider = "OpenAI" # Default

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # 1. API Key Input
    # Use environment variable if available, otherwise let user input
    default_api_key = os.getenv(f"{st.session_state.current_llm_provider.upper()}_API_KEY", "")
    api_key = st.text_input(
        f"Enter {st.session_state.current_llm_provider} API Key",
        type="password",
        value=default_api_key, # Pre-fill if env var exists
        help="Your API key is not stored persistently by the app."
        )

    # 2. LLM Selection
    llm_provider = st.selectbox(
        "Select LLM Provider",
        ("OpenAI", "Anthropic", "Gemini", "DeepSeek"), # Add more as needed
        index=["OpenAI", "Anthropic", "Gemini", "DeepSeek"].index(st.session_state.current_llm_provider), # Default selection
        key="llm_selector" # Add key for potential callbacks later if needed
    )
    # Update session state if selection changes (Streamlit reruns on widget change)
    st.session_state.current_llm_provider = llm_provider


    # 3. File Uploader
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"],
        help="CSV should have a column containing the text chunks (e.g., named 'text', 'chunk', 'content')."
        )

    # Dynamically find text column
    text_column = None
    df_preview_cols = None
    if uploaded_file:
        try:
            # Read just the header to find potential text columns
            # Use BytesIO to avoid issues with multiple reads on the uploaded file object
            bytes_data = uploaded_file.getvalue() # Read file content into memory
            preview_df = pd.read_csv(BytesIO(bytes_data), nrows=0)
            df_preview_cols = preview_df.columns.tolist() # Get all columns

            potential_cols = [col for col in preview_df.columns if any(keyword in col.lower() for keyword in ['text', 'chunk', 'content'])]

            if potential_cols:
                # If only one likely candidate, pre-select it
                default_index = 0
                if len(potential_cols) == 1:
                     # Find index of the single potential col in the full list
                     try:
                         default_index = potential_cols.index(potential_cols[0])
                     except ValueError:
                         default_index = 0 # Should not happen, but fallback

                text_column = st.selectbox("Select Text Column", potential_cols, index=default_index, key="text_col_select")
            elif df_preview_cols: # If no likely candidates found, let user pick from all
                st.warning("Could not automatically detect text column. Please select manually.")
                text_column = st.selectbox("Select Text Column Manually", df_preview_cols, index = 0, key="text_col_select_manual")
            else:
                 st.error("CSV file seems to have no columns. Please check the file.")
                 uploaded_file = None # Invalidate file

        except Exception as e:
            st.error(f"Error reading CSV header: {e}")
            uploaded_file = None # Invalidate file on header read error

    # 4. Processing Button
    process_button = st.button(
        "✨ Extract References",
        disabled=(not api_key or not uploaded_file or not text_column) # Disable if inputs missing
        )

# --- Main Area for Processing and Results ---
if process_button:
    st.session_state.processing_complete = False
    st.session_state.results_data = None
    st.session_state.error_message = None

    # Re-check conditions just before processing
    if not api_key:
        st.error("❌ Please enter your API key.")
    elif not uploaded_file:
        st.error("❌ Please upload a CSV file.")
    elif not text_column:
        st.error("❌ Please select the column containing text chunks.")
    else:
        # Load the full CSV now using the in-memory data
        df = load_csv(BytesIO(bytes_data)) # Pass the bytes data to load_csv

        if df is None:
            st.error("❌ Failed to load the CSV file.")
        elif text_column not in df.columns:
             st.error(f"❌ Selected text column '{text_column}' not found in the loaded CSV. Columns found: {df.columns.tolist()}")
        else:
            st.info(f"Processing {len(df)} text chunks using {llm_provider}...")
            all_extracted_references = []
            total_chunks = len(df)
            progress_bar = st.progress(0)
            status_text = st.empty() # Placeholder for status updates
            start_time = time.time()

            # Use st.expander for potentially long-running logs
            with st.expander("Processing Log", expanded=False):
                st.write(f"Starting reference extraction at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                log_placeholder = st.empty()
                log_messages = []

                for i, row in df.iterrows():
                    current_progress = (i + 1) / total_chunks
                    status_text.info(f"Processing chunk {i+1}/{total_chunks}...")
                    progress_bar.progress(current_progress)

                    chunk = row[text_column]
                    if pd.isna(chunk) or not isinstance(chunk, str) or not chunk.strip():
                        log_msg = f"Chunk {i+1}: Skipped (empty or invalid content)."
                        log_messages.append(log_msg)
                        log_placeholder.markdown(f"`{log_msg}`")
                        # print(log_msg) # Also print to console if running locally
                        continue

                    try:
                        # Call the LLM - returns a list of reference dicts or error dict
                        references_or_error = get_references_from_llm(api_key, llm_provider, chunk)

                        # Check if the result is the error indicator list
                        if isinstance(references_or_error, list) and len(references_or_error) == 1 and isinstance(references_or_error[0], dict) and "Error" in references_or_error[0]:
                            error_detail = references_or_error[0]["Error"]
                            log_msg = f"Chunk {i+1}: API Error - {error_detail}"
                            log_messages.append(log_msg)
                            log_placeholder.markdown(f"`{log_msg}`")
                            # Append error entry to results
                            all_extracted_references.append({
                                "Original Text Chunk": chunk,
                                "Error": error_detail # Include the specific error
                            })
                        elif isinstance(references_or_error, list):
                            ref_count = len(references_or_error)
                            log_msg = f"Chunk {i+1}: Success - Extracted {ref_count} reference(s)."
                            log_messages.append(log_msg)
                            log_placeholder.markdown(f"`{log_msg}`")
                            # Add the original chunk text to each extracted reference
                            for ref in references_or_error:
                               if isinstance(ref, dict): # Ensure it's a dictionary
                                   ref["Original Text Chunk"] = chunk # Add original chunk context
                                   all_extracted_references.append(ref)
                               else:
                                   # Handle unexpected format from LLM gracefully
                                   log_msg_warn = f"Chunk {i+1}: Warning - Invalid item format in LLM response list: {ref}"
                                   log_messages.append(log_msg_warn)
                                   print(log_msg_warn) # Print warning to console too
                                   all_extracted_references.append({
                                       "Original Text Chunk": chunk,
                                       "Error": "Invalid reference format received from LLM"
                                   })
                        else:
                            # Handle case where LLM function returns something unexpected (not a list)
                            log_msg_err = f"Chunk {i+1}: Error - Unexpected return type from LLM function: {type(references_or_error)}"
                            log_messages.append(log_msg_err)
                            log_placeholder.markdown(f"`{log_msg_err}`")
                            all_extracted_references.append({
                                "Original Text Chunk": chunk,
                                "Error": "Internal error: Unexpected data structure from LLM integration."
                            })


                        # Optional: Add a small delay to avoid hitting rate limits
                        # time.sleep(0.2) # Adjust delay as needed

                    except Exception as e:
                        # Catch any other unexpected errors during the loop iteration
                        error_msg = f"🚨 Critical Error processing chunk {i+1}: {e}"
                        st.error(error_msg) # Show prominent error in main area
                        log_messages.append(error_msg)
                        log_placeholder.markdown(f"`{error_msg}`")
                        # Add an error entry to the results
                        all_extracted_references.append({
                            "Original Text Chunk": chunk,
                            "Error": f"App error during processing: {e}"
                        })


            # Finalize progress bar and status text
            progress_bar.progress(1.0)
            status_text.success("Processing loop complete.")
            end_time = time.time()
            st.success(f"✅ Finished processing {total_chunks} chunks in {end_time - start_time:.2f} seconds!")

            if not all_extracted_references:
                st.warning("⚠️ No references were extracted, or all chunks resulted in errors.")
                st.session_state.error_message = "No references extracted or all failed."
                st.session_state.processing_complete = False # Indicate failure/no results
            else:
                 st.session_state.results_data = prepare_output_csv(all_extracted_references)
                 if st.session_state.results_data:
                    st.session_state.processing_complete = True
                 else:
                    st.error("❌ Failed to prepare the output CSV data.")
                    st.session_state.error_message = "Failed to prepare output CSV."
                    st.session_state.processing_complete = False

# --- Display Results and Download Button ---
if st.session_state.processing_complete and st.session_state.results_data:
    st.subheader("📊 Extracted References Preview (First 20 Rows)")
    try:
        # Use BytesIO for reading the prepared CSV bytes data
        results_df_preview = pd.read_csv(BytesIO(st.session_state.results_data), nrows=20)
        st.dataframe(results_df_preview)

        # Generate a safe file name
        base_filename = "extracted_references"
        provider_name = st.session_state.current_llm_provider.lower()
        original_filename = getattr(uploaded_file, 'name', 'inputfile.csv')
        # Basic sanitization for the original filename part
        safe_original_filename = "".join(c for c in original_filename if c.isalnum() or c in ('.', '_', '-')).rstrip()
        download_filename = f"{base_filename}_{provider_name}_{safe_original_filename}"

        st.download_button(
            label="📥 Download All References as CSV",
            data=st.session_state.results_data,
            file_name=download_filename,
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Error displaying preview or preparing download: {e}")
        st.session_state.error_message = f"Error in results display: {e}"

elif st.session_state.error_message:
    st.error(f"Process ended with an error: {st.session_state.error_message}")

# --- Footer or additional info ---
st.markdown("---")
st.markdown("Developed for reference extraction tasks.")
