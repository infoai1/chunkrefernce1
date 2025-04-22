# --- START OF CORRECT FILE chunkrefernce1-main/utils.py ---

import pandas as pd
import streamlit as st # Import streamlit for st.error
from io import StringIO, BytesIO
import sys # For checking max CSV field size limit

# Increase CSV field size limit - useful for very large text chunks
# Be cautious with memory usage if limits are set extremely high
max_int = sys.maxsize
while True:
    # Decrease the max_int value by factor 10
    # as long as the OverflowError occurs.
    try:
        # We need to import csv here specifically for this setting
        import csv
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int/10)

def load_csv(uploaded_file_or_buffer):
    """
    Loads CSV from Streamlit upload object or an in-memory buffer
    (like BytesIO) into a pandas DataFrame.
    Handles potential encoding issues.
    """
    if uploaded_file_or_buffer is None:
        st.error("No file or buffer provided to load_csv.")
        return None

    try:
        # Try reading with standard UTF-8 encoding
        return pd.read_csv(uploaded_file_or_buffer)
    except UnicodeDecodeError:
        try:
            # Fallback to latin1 if utf-8 fails
            if hasattr(uploaded_file_or_buffer, 'seek'):
                uploaded_file_or_buffer.seek(0)
            return pd.read_csv(uploaded_file_or_buffer, encoding='latin1')
        except Exception as e:
            st.error(f"Error reading CSV file with fallback encoding (latin1): {e}")
            return None
    except pd.errors.EmptyDataError:
         st.error("Error: The uploaded CSV file is empty.")
         return None
    except pd.errors.ParserError as e:
         st.error(f"Error: Failed to parse the CSV file. Please ensure it's a valid CSV. Details: {e}")
         # Often caused by inconsistencies in quotes or delimiters, or exceeding field size limit.
         st.info(f"Note: The CSV field size limit is currently set to {max_int}. If you have extremely large text fields, this might be the issue.")
         return None
    except Exception as e:
        st.error(f"Error processing CSV file: {e}")
        return None


def prepare_output_csv(results_list):
    """
    Converts the list of reference dictionaries (potentially including error rows)
    into CSV formatted bytes, ready for download. Ensures specific column order.
    """
    if not results_list:
        st.warning("prepare_output_csv called with empty results list.")
        return None

    try:
        df = pd.DataFrame(results_list)

        # Define the desired order of columns
        output_columns = [
            "Original Text Chunk", # Context added in app.py loop
            "Reference Category",
            "Reference Domain",
            "Reference Detail",
            "Tags",
            "Reference Source Name",
            "Author Mentioned",
            "Is Religious?",
            "Direct Quote",
            "New Term/Concept?",
            "Date or Time Reference",
            "Place or Location",
            "Language of Source",
            "Summary of Relevance",
            "Error" # Include Error column for rows where processing failed
        ]

        # Ensure all desired columns exist, adding missing ones with NaN
        for col in output_columns:
            if col not in df.columns:
                df[col] = pd.NA # Use pandas NA for missing values

        # Reorder DataFrame columns according to the desired list
        df_reordered = df[output_columns]

        # Use BytesIO to create the CSV in memory
        output_buffer = BytesIO()
        # Write to buffer with UTF-8 encoding + BOM for Excel compatibility
        # Use quoting=csv.QUOTE_NONNUMERIC or csv.QUOTE_ALL if fields might contain commas/quotes incorrectly
        import csv # import here just for the quoting constants
        df_reordered.to_csv(output_buffer, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_MINIMAL)

        # Rewind buffer
        output_buffer.seek(0)

        return output_buffer.getvalue() # Return the raw bytes

    except Exception as e:
        st.error(f"Error preparing output CSV: {e}")
        return None

# --- END OF CORRECT FILE chunkrefernce1-main/utils.py ---
