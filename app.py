import streamlit as st
import pandas as pd
from io import StringIO
import time # For potential delays
from dotenv import load_dotenv # Import the function
load_dotenv() # Load variables from .env file into environment
# Import functions from local modules
from llm_integrations import get_references_from_llm
from utils import load_csv, prepare_output_csv




# ... rest of your app.py imports


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

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # 1. API Key Input
    api_key = st.text_input("Enter LLM API Key", type="password", help="Your API key is not stored.")

    # 2. LLM Selection
    llm_provider = st.selectbox(
        "Select LLM Provider",
        ("OpenAI", "Anthropic", "Gemini", "DeepSeek"), # Add more as needed
        index=0 # Default selection
    )

    # 3. File Uploader
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"],
        help="CSV should have a column containing the text chunks (e.g., named 'text', 'chunk', 'content')."
        )

    # Dynamically find text column
    text_column = None
    if uploaded_file:
        try:
            # Read just the header to find potential text columns
            preview_df = pd.read_csv(uploaded_file, nrows=0)
            potential_cols = [col for col in preview_df.columns if 'text' in col.lower() or 'chunk' in col.lower() or 'content' in col.lower()]
            if potential_cols:
                # Let user confirm or select if multiple possibilities exist
                text_column = st.selectbox("Select Text Column", potential_cols, index=0)
            else:
                st.warning("Could not automatically detect text column. Please ensure one exists.")
                text_column = st.selectbox("Select Text Column Manually", preview_df.columns)
            uploaded_file.seek(0) # IMPORTANT: Reset file pointer after reading header
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

    if not api_key:
        st.error("❌ Please enter your API key.")
    elif not uploaded_file:
        st.error("❌ Please upload a CSV file.")
    elif not text_column:
        st.error("❌ Please select the column containing text chunks.")
    else:
        df = load_csv(uploaded_file)
        if df is None or text_column not in df.columns:
            st.error("❌ Failed to load or process the CSV file, or text column not found.")
        else:
            st.info(f"Processing {len(df)} text chunks using {llm_provider}...")
            all_extracted_references = []
            total_chunks = len(df)
            progress_bar = st.progress(0)
            start_time = time.time()

            with st.spinner("Calling LLM API for each chunk... This may take a while."):
                for i, row in df.iterrows():
                    chunk = row[text_column]
                    if pd.isna(chunk) or not isinstance(chunk, str) or not chunk.strip():
                        # Skip empty or non-string chunks
                        progress_bar.progress((i + 1) / total_chunks)
                        continue

                    try:
                        # Call the LLM - returns a list of reference dicts or error dict
                        references = get_references_from_llm(api_key, llm_provider, chunk)

                        # Add the original chunk text to each extracted reference
                        for ref in references:
                           if isinstance(ref, dict): # Ensure it's a dictionary
                               ref["Original Text Chunk"] = chunk # Add original chunk context
                               all_extracted_references.append(ref)
                           else:
                               # Handle unexpected format from LLM gracefully
                               print(f"Warning: Unexpected item format in response list for chunk {i}: {ref}")
                               all_extracted_references.append({
                                   "Original Text Chunk": chunk,
                                   "Error": "Invalid reference format received from LLM"
                               })

                        # Optional: Add a small delay to avoid hitting rate limits
                        # time.sleep(0.5) # Adjust delay as needed

                    except Exception as e:
                        st.error(f"🚨 Error processing chunk {i+1}: {e}")
                        # Add an error entry to the results
                        all_extracted_references.append({
                            "Original Text Chunk": chunk,
                            "Error": f"Failed to process chunk via API: {e}"
                        })

                    # Update progress bar
                    progress_bar.progress((i + 1) / total_chunks)

            end_time = time.time()
            st.success(f"✅ Processing complete in {end_time - start_time:.2f} seconds!")

            if not all_extracted_references:
                st.warning("⚠️ No references were extracted, or all chunks resulted in errors.")
                st.session_state.error_message = "No references extracted."
            else:
                 st.session_state.results_data = prepare_output_csv(all_extracted_references)
                 st.session_state.processing_complete = True

# --- Display Results and Download Button ---
if st.session_state.processing_complete and st.session_state.results_data:
    st.subheader("📊 Extracted References Preview")
    try:
        # Show a preview of the results
        results_df_preview = pd.read_csv(StringIO(st.session_state.results_data.decode('utf-8')), nrows=20)
        st.dataframe(results_df_preview)
    except Exception as e:
        st.error(f"Error displaying preview: {e}")

    st.download_button(
        label="📥 Download All References as CSV",
        data=st.session_state.results_data,
        file_name=f"extracted_references_{llm_provider}_{uploaded_file.name}",
        mime="text/csv",
    )
elif st.session_state.error_message:
    st.error(st.session_state.error_message)
