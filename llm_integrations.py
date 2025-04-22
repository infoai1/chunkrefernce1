# --- START OF FILE chunkrefernce1-main/llm_integrations.py ---

import requests
import os
import json
# Import specific SDKs if used
# Note: We ARE using the 'openai' SDK below, make sure it's installed!
# from anthropic import Anthropic
# import google.generativeai as genai

# --- Define the Detailed Prompt ---
# Incorporate ALL the columns you decided on
DETAILED_PROMPT_TEMPLATE = """
You are an expert reference extractor. Analyze the text chunk provided below.
Extract ALL meaningful references according to the schema described.
For EACH reference found, return a JSON object with the following keys:
"Reference Category", "Reference Domain", "Reference Detail", "Tags", "Reference Source Name",
"Author Mentioned", "Is Religious?", "Direct Quote", "New Term/Concept?",
"Date or Time Reference", "Place or Location", "Language of Source", "Summary of Relevance".

SCHEMA DETAILS:
- Reference Category: Specific type (e.g., Quran Verse, Hadith, Historical Event, Scientific Fact, Poetry Line, Dream, New Term by Author, Quote from Person, etc.)
- Reference Domain: Broad category (Religious & Scriptural, Historical & Biographical, Literary & Cultural, Academic & Intellectual, Personal & Experiential, Scientific & Technical, Social, Economic & Political)
- Reference Detail: Precise description (e.g., "Quran 2:255 (Ayat al-Kursi)", "Hadith on Patience from Sahih Bukhari", "1947 Partition of India", "Concept of 'Positive Thinking' as defined by author")
- Tags: 3-5 concise keywords (e.g., ["Quran", "Tawhid", "Ayat al-Kursi"], ["Hadith", "Patience", "Bukhari"])
- Reference Source Name: Specific text/book/scholar/person cited (e.g., "Sahih Bukhari", "Quran", "Author's observation", "Ihya Ulum al-Din")
- Author Mentioned: Name of person mentioned, if any (e.g., "Prophet Muhammad", "Ghazali", "None")
- Is Religious?: Boolean true/false.
- Direct Quote: The exact quoted text, if present (e.g., "'The strong is not the one who overcomes...'"). Empty string if none.
- New Term/Concept?: The specific term/concept if unique to the author. Empty string if none.
- Date or Time Reference: Extracted date/year/era (e.g., "7th century", "1988", "None").
- Place or Location: Extracted place (e.g., "Mecca", "India", "None").
- Language of Source: Original language if known (e.g., "Arabic", "Urdu", "English", "Unknown").
- Summary of Relevance: 1-2 sentences on how the reference is used in the chunk.

Output Format: Return ONLY a valid JSON list containing one JSON object for each reference found. If no references are found, return an empty list [].

Text Chunk to Analyze:
---
{text_chunk}
---

JSON Output:
"""

# --- Function to call OpenAI ---
def call_openai_api(api_key, text_chunk, model="gpt-4o"): # <<< FIXED: Changed model to "gpt-4o" (a valid, current model)
    # Requires 'openai' library: pip install openai
    # Make sure 'openai' is listed in your requirements.txt file and installed!
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    prompt = DETAILED_PROMPT_TEMPLATE.format(text_chunk=text_chunk)
    try:
        response = client.chat.completions.create(
            model=model, # Use the model specified in the function definition (now defaults to gpt-4o)
            messages=[
                {"role": "system", "content": "You are an expert reference extractor outputting JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"} # Request JSON output
        )
        # Adjust based on actual response structure - it might be nested
        content = response.choices[0].message.content
        # The LLM might wrap the list in a top-level key, e.g. {"references": [...]}. Adjust parsing accordingly.
        # Or it might just return the list string directly.
        try:
            # Assuming the content IS the JSON list string, possibly inside a wrapper
            # Find the start '[' and end ']' if necessary to handle non-perfect JSON output
            start_index = content.find('[')
            end_index = content.rfind(']')
            if start_index != -1 and end_index != -1:
                 json_list_str = content[start_index:end_index+1]
                 return json.loads(json_list_str)
            else: # Handle cases where it didn't return a list
                 print(f"Warning: OpenAI response did not seem to contain a JSON list: {content}")
                 # Attempt to parse the whole content directly if no list brackets found, maybe JSON format is strict
                 try:
                     parsed_json = json.loads(content)
                     # If the response is {"references": [...]}, extract the list
                     if isinstance(parsed_json, dict) and len(parsed_json) == 1:
                         key = list(parsed_json.keys())[0]
                         if isinstance(parsed_json[key], list):
                             return parsed_json[key]
                     elif isinstance(parsed_json, list): # If it's already a list
                         return parsed_json
                     else: # Unknown format
                         print(f"Warning: Parsed JSON is not a recognized list or wrapper: {parsed_json}")
                         return [] # Return empty list on format error
                 except json.JSONDecodeError:
                     print(f"Warning: OpenAI content could not be parsed as JSON directly: {content}")
                     return [] # Return empty list on format error

        except json.JSONDecodeError:
            print(f"Error: OpenAI response substring was not valid JSON: {json_list_str}")
            return [] # Return empty list on decode error
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        # Check if the error message is about an invalid model
        if "does not exist" in str(e) or "Invalid model" in str(e):
            print(f"---> Critical Error: The selected model '{model}' is likely invalid or unavailable. Check model name.")
        raise # Re-raise the exception to be caught in the main app

# --- Function to call Anthropic ---
def call_anthropic_api(api_key, text_chunk, model="claude-3-opus-20240229"):
    # Requires 'anthropic' library: pip install anthropic
    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)
    prompt = DETAILED_PROMPT_TEMPLATE.format(text_chunk=text_chunk)
    try:
        message = client.messages.create(
            model=model,
            max_tokens=3000, # Increased max_tokens slightly for potentially complex JSON
            system="You are an expert reference extractor outputting ONLY a valid JSON list.", # System prompt
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        content = message.content[0].text
        # Similar parsing logic as OpenAI, find '[' and ']'
        start_index = content.find('[')
        end_index = content.rfind(']')
        if start_index != -1 and end_index != -1:
             json_list_str = content[start_index:end_index+1]
             try:
                 return json.loads(json_list_str)
             except json.JSONDecodeError:
                print(f"Error: Anthropic response substring was not valid JSON: {json_list_str}")
                return []
        else:
             print(f"Warning: Anthropic response did not seem to contain a JSON list: {content}")
             return []
    except Exception as e:
        print(f"Error calling Anthropic API: {e}")
        raise

# --- Function to call Gemini ---
def call_gemini_api(api_key, text_chunk, model="gemini-pro"): # Or newer models like gemini-1.5-pro-latest
    # Requires 'google-generativeai' library: pip install google-generativeai
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    prompt = DETAILED_PROMPT_TEMPLATE.format(text_chunk=text_chunk)
    llm = genai.GenerativeModel(model)
    # Gemini might need specific config for JSON output
    generation_config = genai.types.GenerationConfig(
        # candidate_count=1, # Default
        # max_output_tokens=2048, # Default
        response_mime_type="application/json" # Explicitly ask for JSON
    )
    safety_settings = [ # Relax safety settings slightly if needed, use with caution
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    try:
        response = llm.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
            )
        # Accessing response might differ slightly based on SDK version
        content = response.text
        # No need to find '[' and ']' if response_mime_type works correctly
        return json.loads(content)
    except json.JSONDecodeError:
         print(f"Error: Gemini response was not valid JSON despite requesting JSON MIME type: {content}")
         # Fallback: try finding JSON list within the text if mime_type failed
         start_index = content.find('[')
         end_index = content.rfind(']')
         if start_index != -1 and end_index != -1:
             json_list_str = content[start_index:end_index+1]
             try:
                 return json.loads(json_list_str)
             except json.JSONDecodeError:
                 print(f"Fallback JSON parsing also failed for Gemini.")
                 return []
         return []
    except ValueError as ve:
        # Specific check for content filtering / safety blocks
        if "response was blocked" in str(ve):
             print(f"Warning: Gemini API call blocked due to safety settings. Prompt Feedback: {response.prompt_feedback}")
             return [{"Error": "Content blocked by Gemini safety filters."}]
        else:
             print(f"Error calling Gemini API (ValueError): {ve}")
             raise
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        # print(response.prompt_feedback) # Uncomment to see feedback if available
        raise

# --- Function to call DeepSeek ---
def call_deepseek_api(api_key, text_chunk, model="deepseek-chat"): # Check their specific model names
    # Uses 'requests' library
    API_URL = "https://api.deepseek.com/chat/completions" # VERIFY THIS URL from DeepSeek documentation
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    prompt = DETAILED_PROMPT_TEMPLATE.format(text_chunk=text_chunk)
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert reference extractor outputting ONLY a valid JSON list."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 3000, # Add max_tokens if needed/supported
        # Check DeepSeek docs for JSON mode or similar parameters
    }
    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=90) # Increased timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        response_json = response.json()
        # Navigate the response structure to get the content - CHECK DEEPSEEK DOCS for exact structure
        content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Parsing logic - assuming content should be the JSON list string
        start_index = content.find('[')
        end_index = content.rfind(']')
        if start_index != -1 and end_index != -1:
             json_list_str = content[start_index:end_index+1]
             try:
                 return json.loads(json_list_str)
             except json.JSONDecodeError:
                print(f"Error: DeepSeek response substring was not valid JSON: {json_list_str}")
                return []
        else:
             print(f"Warning: DeepSeek response did not seem to contain a JSON list: {content}")
             return []
    except requests.exceptions.RequestException as e:
        print(f"Error calling DeepSeek API (Network/HTTP): {e}")
        # If it's a 4xx/5xx error, print response body for more details if available
        if e.response is not None:
            print(f"DeepSeek Response Status Code: {e.response.status_code}")
            print(f"DeepSeek Response Body: {e.response.text}")
        raise
    except json.JSONDecodeError:
        # This might happen if response.json() fails above
        print(f"Error: DeepSeek raw response could not be parsed as JSON: {response.text}")
        return []
    except Exception as e:
        print(f"Error processing DeepSeek response: {e}")
        raise


# --- Central Dispatcher ---
def get_references_from_llm(api_key, llm_provider, text_chunk):
    """Calls the appropriate LLM API based on the provider."""
    try:
        if llm_provider == "OpenAI":
            # Use the corrected call_openai_api function
            return call_openai_api(api_key, text_chunk)
        elif llm_provider == "Anthropic":
            return call_anthropic_api(api_key, text_chunk)
        elif llm_provider == "Gemini":
            return call_gemini_api(api_key, text_chunk)
        elif llm_provider == "DeepSeek":
            return call_deepseek_api(api_key, text_chunk)
        else:
            # Should not happen if Streamlit selectbox is used correctly
            print(f"Error: Invalid LLM Provider '{llm_provider}' received.")
            return [{"Error": f"Invalid LLM Provider selected: {llm_provider}"}]
    except Exception as e:
        # Log the error and the chunk causing it for debugging
        # Use repr(e) to potentially get more details about the exception type and message
        print(f"Error processing chunk with {llm_provider}: {text_chunk[:100]}... \nError Type: {type(e).__name__}, Message: {repr(e)}")
        # Return an error dictionary that can be included in the output CSV
        return [{"Error": f"API call failed for provider {llm_provider}. Type: {type(e).__name__}. Check logs or console for details."}]

# --- END OF FILE chunkrefernce1-main/llm_integrations.py ---
