# --- START OF FILE chunkrefernce1-main/app.py ---

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
    # Default to OpenAI API key if provider changes and specific key isn't set
    provider_env_var = f"{st.session_state.current_llm_provider.upper()}_API_KEY"
    default_api_key = os.getenv(provider_env_var, os.getenv("OPENAI_API_KEY", "")) # Fallback slightly more robust

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
        key="llm_selector"
    )
    # Update session state if selection changes
    if llm_provider != st.session_state.current_llm_provider:
        st.session_state.current_llm_provider = llm_provider
        # Clear previous results if provider changes (optional, but good practice)
        st.session_state.processing_complete = False
        st.session_state.results_data = None
        st.session_state.error_message = None
        st.rerun() # Rerun to update API key input label and default value


    # 3. File Uploader
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"],
        help="CSV should have a column containing the text chunks (e.g., named 'text', 'chunk', 'content')."
        )

    # Dynamically find text column
    text_column = None
    df_preview_cols = None
    bytes_data = None # Initialize bytes_data here
    preview_df = None # Initialize preview_df

    if uploaded_file:
        try:
            # Read file content into memory ONCE
            bytes_data = uploaded_file.getvalue()
            # Read just the header to find potential text columns
            preview_df = pd.read_csv(BytesIO(bytes_data), nrows=0) # Use BytesIO for preview
            df_preview_cols = preview_df.columns.tolist()

            potential_cols = [col for col in df_preview_cols if any(keyword in col.lower() for keyword in ['text', 'chunk', 'content'])]

            default_index = 0
            if potential_cols:
                # Try to find exact matches first
                exact_matches = [col for col in potential_cols if col.lower() in ['text', 'chunk', 'content']]
                if exact_matches:
                    # If multiple exact matches, pick the first one
                    selected_col = exact_matches[0]
                elif len(potential_cols) == 1:
                    # If only one potential partial match, use it
                    selected_col = potential_cols[0]
                else:
                    # If multiple partial matches, let user choose, default to first
                    selected_col = potential_cols[0] # Keep first as default

                try:
                    # Find the index in the *original* list for the selectbox
                    default_index = df_preview_cols.index(selected_col)
                except ValueError:
                    default_index = 0 # Fallback

                # Use all columns as options, but set the best guess as default
                text_column = st.selectbox(
                    "Select Text Column",
                    df_preview_cols, # Show all columns
                    index=default_index, # Pre-select best guess
                    key="text_col_select"
                    )
            elif df_preview_cols:
                st.warning("Could not automatically detect text column. Please select manually.")
                text_column = st.selectbox(
                    "Select Text Column Manually",
                    df_preview_cols,
                    index = 0,
                    key="text_col_select_manual"
                    )
            else:
                 st.error("CSV file seems to have no columns. Please check the file.")
                 uploaded_file = None
                 bytes_data = None # Invalidate data

        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty.")
            uploaded_file = None
            bytes_data = None
        except Exception as e:
            st.error(f"Error reading CSV header: {e}")
            uploaded_file = None
            bytes_data = None

    # 4. Processing Button
    process_button = st.button(
        "✨ Extract References",
        disabled=(not api_key or not uploaded_file or not text_column or not bytes_data) # Disable if inputs missing or data invalid
        )

# --- Main Area for Processing and Results ---
if process_button:
    st.session_state.processing_complete = False
    st.session_state.results_data = None
    st.session_state.error_message = None

    # Re-check conditions just before processing
    if not api_key:
        st.error("❌ Please enter your API key.")
    elif not uploaded_file or not bytes_data:
        st.error("❌ Please upload a valid CSV file.")
    elif not text_column:
        st.error("❌ Please select the column containing text chunks.")
    else:
        # Load the full CSV now using the in-memory data
        # Ensure bytes_data is valid before passing
        if bytes_data:
            df = load_csv(BytesIO(bytes_data)) # Pass the bytes data to load_csv
        else:
            df = None # Should not happen if button was enabled, but safety check

        if df is None:
            st.error("❌ Failed to load the CSV file data for processing.")
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
            with st.expander("Processing Log", expanded=True): # Expand log by default for debugging
                st.write(f"Starting reference extraction at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                log_placeholder = st.empty()
                log_messages = []

                for i, row in df.iterrows():
                    current_progress = (i + 1) / total_chunks
                    chunk_processed = False # Flag to ensure something is added for every row

                    # Ensure text column exists in the row before accessing
                    original_chunk_text_for_output = f"Row {i+1}: Error - Invalid row data"
                    if text_column in row:
                        original_chunk_text_for_output = row[text_column]
                    else:
                        log_msg = f"Chunk {i+1}: Error - Text column '{text_column}' missing in this row. Skipping API call."
                        log_messages.append(log_msg)
                        log_placeholder.markdown(f"`{'<br>'.join(log_messages)}`", unsafe_allow_html=True) # Update log display
                        all_extracted_references.append({
                           "Original Text Chunk": f"Error: Column '{text_column}' not found in row {i+1}",
                           "Error": f"Column '{text_column}' missing"
                        })
                        chunk_processed = True # Mark as processed (error handled)
                        progress_bar.progress(current_progress) # Update progress bar
                        status_text.info(f"Processing chunk {i+1}/{total_chunks}... (Skipped: Missing Column)")
                        continue # Skip API call for this row

                    chunk = original_chunk_text_for_output # Use the validated chunk text
                    status_text.info(f"Processing chunk {i+1}/{total_chunks}...")
                    progress_bar.progress(current_progress)


                    # Check for valid chunk content BEFORE calling LLM
                    if pd.isna(chunk) or not isinstance(chunk, str) or not chunk.strip():
                        log_msg = f"Chunk {i+1}: Skipped (empty or invalid content)."
                        log_messages.append(log_msg)
                        log_placeholder.markdown(f"`{'<br>'.join(log_messages)}`", unsafe_allow_html=True) # Update log display
                        all_extracted_references.append({
                            "Original Text Chunk": str(chunk)[:500] + "..." if isinstance(chunk, str) and len(chunk)>500 else str(chunk),
                            "Reference Detail": "Skipped - Empty or invalid content"
                        })
                        chunk_processed = True
                        continue

                    try:
                        # Call the LLM - returns a list of reference dicts or error dict
                        references_or_error = get_references_from_llm(api_key, llm_provider, chunk)

                        # Check if the result is the error indicator list
                        if isinstance(references_or_error, list) and len(references_or_error) == 1 and isinstance(references_or_error[0], dict) and "Error" in references_or_error[0]:
                            error_detail = references_or_error[0]["Error"]
                            log_msg = f"Chunk {i+1}: API/Processing Error - {error_detail}"
                            log_messages.append(log_msg)
                            log_placeholder.markdown(f"`{'<br>'.join(log_messages)}`", unsafe_allow_html=True)
                            # Append error entry to results
                            all_extracted_references.append({
                                "Original Text Chunk": chunk,
                                "Error": error_detail # Include the specific error
                            })
                            chunk_processed = True
                        elif isinstance(references_or_error, list):
                            # Check if the list is empty
                            if not references_or_error:
                                # No references found by LLM
                                log_msg = f"Chunk {i+1}: Success - No references found."
                                log_messages.append(log_msg)
                                log_placeholder.markdown(f"`{'<br>'.join(log_messages)}`", unsafe_allow_html=True)
                                all_extracted_references.append({
                                    "Original Text Chunk": chunk,
                                    "Reference Detail": "No references found" # Indicate status clearly
                                })
                                chunk_processed = True
                            else:
                                # References were found (original logic)
                                ref_count = len(references_or_error)
                                log_msg = f"Chunk {i+1}: Success - Extracted {ref_count} reference(s)."
                                log_messages.append(log_msg)
                                log_placeholder.markdown(f"`{'<br>'.join(log_messages)}`", unsafe_allow_html=True)
                                # Add the original chunk text to each extracted reference
                                for ref in references_or_error:
                                   if isinstance(ref, dict): # Ensure it's a dictionary
                                       ref["Original Text Chunk"] = chunk # Add original chunk context
                                       all_extracted_references.append(ref)
                                   else:
                                       # Handle unexpected item format from LLM gracefully
                                       log_msg_warn = f"Chunk {i+1}: Warning - Invalid item format in LLM response list: {ref}"
                                       log_messages.append(log_msg_warn)
                                       print(log_msg_warn) # Print warning to console too
                                       all_extracted_references.append({
                                           "Original Text Chunk": chunk,
                                           "Error": "Invalid reference format received from LLM",
                                           "Reference Detail": f"Raw invalid item: {str(ref)[:100]}" # Include part of invalid data
                                       })
                                chunk_processed = True # Mark processed even if some items were invalid
                        else:
                            # Handle case where LLM function returns something unexpected (not a list)
                            log_msg_err = f"Chunk {i+1}: Error - Unexpected return type from LLM function: {type(references_or_error)}"
                            log_messages.append(log_msg_err)
                            log_placeholder.markdown(f"`{'<br>'.join(log_messages)}`", unsafe_allow_html=True)
                            all_extracted_references.append({
                                "Original Text Chunk": chunk,
                                "Error": "Internal error: Unexpected data structure from LLM integration."
                            })
                            chunk_processed = True


                        # Optional: Add a small delay to avoid hitting rate limits
                        # time.sleep(0.2) # Adjust delay as needed

                    except Exception as e:
                        # Catch any other unexpected errors during the loop iteration
                        error_msg = f"🚨 Critical Error processing chunk {i+1}: {e}"
                        st.error(error_msg) # Show prominent error in main area
                        log_messages.append(error_msg)
                        log_placeholder.markdown(f"`{'<br>'.join(log_messages)}`", unsafe_allow_html=True)
                        # Add an error entry to the results
                        all_extracted_references.append({
                            "Original Text Chunk": chunk,
                            "Error": f"App error during processing: {e}"
                        })
                        chunk_processed = True

                    # Add a placeholder if nothing else was added for this row (safety net)
                    if not chunk_processed:
                         log_msg_unhandled = f"Chunk {i+1}: Unhandled state - adding placeholder error."
                         log_messages.append(log_msg_unhandled)
                         log_placeholder.markdown(f"`{'<br>'.join(log_messages)}`", unsafe_allow_html=True)
                         all_extracted_references.append({
                            "Original Text Chunk": chunk,
                            "Error": "Unhandled processing state"
                         })


            # Finalize progress bar and status text
            progress_bar.progress(1.0)
            status_text.success("Processing loop complete.")
            end_time = time.time()
            st.success(f"✅ Finished processing {total_chunks} chunks in {end_time - start_time:.2f} seconds!")

            # Check if ANY data was collected (including errors or 'no reference' rows)
            if not all_extracted_references:
                st.warning("⚠️ No references were extracted, and no errors or skipped chunks were recorded.")
                st.session_state.error_message = "Processing completed, but no data was generated."
                st.session_state.processing_complete = False
            else:
                 # Ensure results_data is generated even if only errors/no-ref rows exist
                 results_bytes = prepare_output_csv(all_extracted_references)
                 if results_bytes:
                    st.session_state.results_data = results_bytes # Store bytes directly
                    st.session_state.processing_complete = True
                 else:
                    st.error("❌ Failed to prepare the output CSV data.")
                    st.session_state.error_message = "Failed to prepare output CSV."
                    st.session_state.processing_complete = False

# --- Display Results and Download Button ---
# Modified check to handle bytes directly
if st.session_state.processing_complete and st.session_state.results_data:
    st.subheader("📊 Processing Results Preview (First 20 Rows)")
    try:
        # Use BytesIO for reading the prepared CSV bytes data
        results_df_preview = pd.read_csv(BytesIO(st.session_state.results_data), nrows=20)
        st.dataframe(results_df_preview)

        # Generate a safe file name
        base_filename = "extracted_references"
        provider_name = st.session_state.current_llm_provider.lower()
        # Use uploaded_file.name if available, otherwise fallback
        original_filename = uploaded_file.name if uploaded_file else 'inputfile.csv'
        # Basic sanitization
        safe_original_filename = "".join(c for c in original_filename if c.isalnum() or c in ('.', '_', '-')).rstrip('.csv') # Prevent multiple .csv
        download_filename = f"{base_filename}_{provider_name}_{safe_original_filename}.csv"

        st.download_button(
            label="📥 Download All Results as CSV",
            data=st.session_state.results_data, # Pass the bytes
            file_name=download_filename,
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Error displaying preview or preparing download: {e}")
        st.session_state.error_message = f"Error in results display: {e}"
        # Optionally reset state if display fails critically
        # st.session_state.processing_complete = False
        # st.session_state.results_data = None

elif st.session_state.error_message:
    st.error(f"Process ended with an error or issue: {st.session_state.error_message}")

# --- Footer or additional info ---
st.markdown("---")
st.markdown("Developed for reference extraction tasks.")

# --- END OF FILE chunkrefernce1-main/app.py ---
