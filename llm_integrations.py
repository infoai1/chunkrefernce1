import pandas as pd
import streamlit as st # Import streamlit for st.error
from io import StringIO, BytesIO

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
        # The input might be a Streamlit UploadedFile or a BytesIO buffer
        # Both should be readable by pd.read_csv directly
        return pd.read_csv(uploaded_file_or_buffer)
    except UnicodeDecodeError:
        try:
            # Fallback to latin1 if utf-8 fails
            # Need to reset the buffer's pointer if it's an IO object
            if hasattr(uploaded_file_or_buffer, 'seek'):
                uploaded_file_or_buffer.seek(0)
            return pd.read_csv(uploaded_file_or_buffer, encoding='latin1')
        except Exception as e:
            st.error(f"Error reading CSV file with fallback encoding: {e}")
            return None
    except pd.errors.EmptyDataError:
         st.error("Error: The uploaded CSV file is empty.")
         return None
    except pd.errors.ParserError:
         st.error("Error: Failed to parse the CSV file. Please ensure it's a valid CSV.")
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

        # Define the desired order of columns, matching the prompt's request + Error + Context
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

        # Use BytesIO to create the CSV in memory without writing to disk
        output_buffer = BytesIO()
        # Write to buffer with UTF-8 encoding (standard for CSV)
        df_reordered.to_csv(output_buffer, index=False, encoding='utf-8-sig') # utf-8-sig includes BOM for Excel compatibility

        # Rewind buffer to the beginning so its content can be read
        output_buffer.seek(0)

        return output_buffer.getvalue() # Return the raw bytes

    except Exception as e:
        st.error(f"Error preparing output CSV: {e}")
        # Optionally log the results_list that caused the error for debugging
        # print("Data causing CSV preparation error:", results_list)
        return None
