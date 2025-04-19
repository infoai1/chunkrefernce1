import pandas as pd
from io import StringIO, BytesIO

def load_csv(uploaded_file):
    """Loads CSV from Streamlit upload into a pandas DataFrame."""
    if uploaded_file is not None:
        try:
            # Try reading with standard encoding
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            try:
                # Fallback to latin1 if utf-8 fails
                uploaded_file.seek(0) # Reset file pointer
                return pd.read_csv(uploaded_file, encoding='latin1')
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                return None
        except Exception as e:
            st.error(f"Error processing CSV file: {e}")
            return None
    return None

def prepare_output_csv(results_list):
    """Converts the list of reference dictionaries to a CSV string."""
    if not results_list:
        return None

    df = pd.DataFrame(results_list)

    # Define the desired order of columns, matching the prompt's request
    output_columns = [
        "Original Text Chunk", "Reference Category", "Reference Domain",
        "Reference Detail", "Tags", "Reference Source Name", "Author Mentioned",
        "Is Religious?", "Direct Quote", "New Term/Concept?",
        "Date or Time Reference", "Place or Location", "Language of Source",
        "Summary of Relevance", "Error" # Include Error column if you add error indicators
    ]

    # Reorder DataFrame columns, adding missing ones with None/NaN
    df_reordered = df.reindex(columns=output_columns)


    output = BytesIO()
    df_reordered.to_csv(output, index=False, encoding='utf-8')
    output.seek(0) # Rewind buffer
    return output.getvalue()

# Potentially add parsing functions here if needed, although the API calls try to parse directly now.
