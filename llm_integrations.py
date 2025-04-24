# --- START OF FILE chunkrefernce1-main/llm_integrations.py ---

import requests
import os
import json
import time # Potentially useful for adding delays

# Import specific SDKs ONLY if used (make sure they are in requirements.txt)
# from anthropic import Anthropic
# import google.generativeai as genai
# Make sure openai library is installed (listed in requirements.txt)
try:
    from openai import OpenAI # We are using this one
except ImportError:
    # This allows the file to be imported even if openai isn't installed yet,
    # but the function call will fail later if it's missing.
    print("WARNING: OpenAI library not found. Install with 'pip install openai'")
    OpenAI = None


# --- Define the Detailed Prompt ---
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
- Direct Quote: The exact quoted text, if present (e.g., "'The strong is not the one who overcomes...'"). Return empty string "" if none.
- New Term/Concept?: The specific term/concept if unique to the author. Return empty string "" if none.
- Date or Time Reference: Extracted date/year/era (e.g., "7th century", "1988"). Return "None" as a string if none.
- Place or Location: Extracted place (e.g., "Mecca", "India"). Return "None" as a string if none.
- Language of Source: Original language if known (e.g., "Arabic", "Urdu", "English"). Return "Unknown" if not determinable.
- Summary of Relevance: 1-2 sentences on how the reference is used in the chunk.

Output Format: Your response MUST be a single JSON object. Inside this object, there MUST be a key (e.g., "references") whose value is a JSON list containing one JSON object for each reference found. If no references are found, the value should be an empty list []. Example: {"references": [{"Reference Category": "..."}, {"Reference Category": "..."}]} or {"references": []}. Do not include any explanatory text before or after the JSON object.

Text Chunk to Analyze:
---
{text_chunk}
---

JSON Output:
"""

# --- Function to call OpenAI ---
def call_openai_api(api_key, text_chunk): # No model default here
    if OpenAI is None:
        return [{"Error": "OpenAI library not installed. Please install it."}]

    # --- OpenAI GPT Configuration ---
    # Explicitly set the desired model (ensure this is the correct identifier)
    model_name = "gpt-4.1" # CHANGED FROM -4o default/env var
    # Note: Verify "gpt-4-turbo" is the current correct name for the model you intend.
    # OpenAI model names can change (e.g., gpt-4-0125-preview, etc.)

    print(f"--- Attempting OpenAI API call with model: {model_name} ---")

    try:
        client = OpenAI(api_key=api_key)
        prompt = DETAILED_PROMPT_TEMPLATE.format(text_chunk=text_chunk)

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert reference extractor outputting ONLY a valid JSON object containing a list, as described."}, # Updated system prompt slightly
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}, # Keep this - it guarantees valid JSON object output
            temperature=0.2,
            max_tokens=3000 # Consider increasing if chunks are very long or references complex
        )

        content = response.choices[0].message.content

        try:
            # Parse the guaranteed JSON object
            parsed_json = json.loads(content)

            # *** MODIFIED JSON HANDLING LOGIC ***
            # Check if it's a dictionary, then look for *any* value that is a list.
            if isinstance(parsed_json, dict):
                found_list = None
                for key, value in parsed_json.items():
                    if isinstance(value, list):
                        found_list = value
                        print(f"--- OpenAI Success: Found list under key '{key}'. Items: {len(found_list)} ---")
                        break # Stop searching once a list is found
                if found_list is not None:
                    return found_list # Return the found list directly
                else:
                    # It's a dict, but no value was a list.
                    print(f"Warning: OpenAI response was a JSON object but contained no list value: {content}")
                    return [{"Error": "LLM returned JSON object, but no list found inside."}]
            # *** END MODIFIED LOGIC ***
            # Fallback: Should not happen with response_format=json_object, but check just in case.
            elif isinstance(parsed_json, list):
                 print(f"--- OpenAI Success: Parsed list directly (unexpected with json_object format). Items: {len(parsed_json)} ---")
                 return parsed_json
            else:
                 # It parsed as JSON, but wasn't a dict or list? Highly unlikely.
                 print(f"Warning: OpenAI response was valid JSON but not a dict or list: {content}")
                 return [{"Error": "LLM returned unexpected JSON structure (not dict/list)."}]

        except json.JSONDecodeError:
            # This error should be much less likely now with response_format={"type": "json_object"}
            print(f"Error: OpenAI response could not be parsed as JSON despite json_object request: {content[:200]}...")
            # Add bracket search back as a last resort, though it shouldn't be needed.
            start_index = content.find('[')
            end_index = content.rfind(']')
            if start_index != -1 and end_index != -1 and start_index < end_index:
                 json_list_str = content[start_index : end_index + 1]
                 try:
                     parsed_list = json.loads(json_list_str)
                     print(f"--- OpenAI Success: Parsed list via bracket search (fallback). Items: {len(parsed_list)} ---")
                     return parsed_list
                 except json.JSONDecodeError:
                     print(f"Error: Fallback bracket search also failed to parse JSON: {json_list_str[:200]}...")
                     return [{"Error": "LLM response could not be parsed as JSON list (fallback failed)."}]
            else:
                print(f"Error: Could not find JSON list brackets '[]' in response (fallback check): {content[:200]}...")
                return [{"Error": "LLM did not return a JSON list format (fallback check)."}]

    except Exception as e:
        error_message = f"Error calling OpenAI API: {e}"
        print(f"!!! {error_message} !!!")
        # Keep the specific model error check
        if "does not exist" in str(e) or "Invalid model" in str(e) or "model_not_found" in str(e):
            error_detail = f"The specified OpenAI model '{model_name}' was not found or is invalid. Check the model name in llm_integrations.py. Original error: {e}"
            print(f"!!! Critical Error: {error_detail} !!!")
            return [{"Error": error_detail}]
        # General API error
        return [{"Error": f"OpenAI API call failed: {e}"}]


# --- Function to call Anthropic ---
# (Keep Anthropic, Gemini, DeepSeek functions as they were - no changes needed there)
def call_anthropic_api(api_key, text_chunk, model="claude-3-opus-20240229"):
    try:
        from anthropic import Anthropic
        print(f"--- Attempting Anthropic API call with model: {model} ---")
        client = Anthropic(api_key=api_key)
        prompt = DETAILED_PROMPT_TEMPLATE.format(text_chunk=text_chunk).replace('"references": ', '') # Remove wrapper hint for Anthropic if needed
        message = client.messages.create(
             model=model, max_tokens=3500, system="Respond ONLY with valid JSON list.",
             messages=[{"role": "user", "content": prompt}]
             )
        content = message.content[0].text
        start_index = content.find('[')
        end_index = content.rfind(']')
        if start_index != -1 and end_index != -1 and start_index < end_index:
             json_list_str = content[start_index : end_index + 1]
             try: return json.loads(json_list_str)
             except json.JSONDecodeError: return [{"Error": "LLM response could not be parsed as JSON list (Anthropic)."}]
        else: return [{"Error": "LLM did not return a JSON list format (Anthropic)."}]
    except ImportError: return [{"Error": "Anthropic library not installed."}]
    except Exception as e: return [{"Error": f"Anthropic API call failed: {e}"}]


# --- Function to call Gemini ---
def call_gemini_api(api_key, text_chunk, model="gemini-1.5-pro-latest"):
    try:
        import google.generativeai as genai
        print(f"--- Attempting Gemini API call with model: {model} ---")
        genai.configure(api_key=api_key)
        prompt = DETAILED_PROMPT_TEMPLATE.format(text_chunk=text_chunk).replace('"references": ', '') # Remove wrapper hint for Gemini if needed
        generation_config = genai.GenerationConfig(response_mime_type="application/json", temperature=0.2)
        safety_settings=[{"category":c,"threshold":"BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT","HARM_CATEGORY_HATE_SPEECH","HARM_CATEGORY_SEXUALLY_EXPLICIT","HARM_CATEGORY_DANGEROUS_CONTENT"]]
        llm = genai.GenerativeModel(model_name=model, generation_config=generation_config, safety_settings=safety_settings)
        response = llm.generate_content(prompt)
        content = response.text
        try:
             parsed_json = json.loads(content)
             if isinstance(parsed_json, list): return parsed_json
             elif isinstance(parsed_json, dict): # Check wrapped object (common Gemini behavior)
                  for key, value in parsed_json.items():
                      if isinstance(value, list):
                          return value
             return [{"Error": "LLM returned unexpected JSON structure (Gemini)."}]
        except json.JSONDecodeError: return [{"Error": "LLM response could not be parsed as JSON list (Gemini)."}]
        except ValueError as ve: # Safety blocking
             if "response was blocked" in str(ve): return [{"Error": f"Content blocked by Gemini safety filters."}]
             else: raise ve # Re-raise other ValueErrors
    except ImportError: return [{"Error": "Google GenerativeAI library not installed."}]
    except Exception as e: return [{"Error": f"Gemini API call failed: {e}"}]


# --- Function to call DeepSeek ---
def call_deepseek_api(api_key, text_chunk, model="deepseek-chat"):
    try:
        API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions")
        print(f"--- Attempting DeepSeek API call with model: {model} ---")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        prompt_content = DETAILED_PROMPT_TEMPLATE.format(text_chunk=text_chunk).replace('"references": ', '') # Remove wrapper hint for DeepSeek if needed
        data = {"model": model, "messages": [{"role":"system", "content":"Respond ONLY with valid JSON list."},{"role":"user","content":prompt_content}], "max_tokens":3500, "temperature":0.2, "response_format": {"type": "json_object"}} # Add DeepSeek json format if supported
        response = requests.post(API_URL, headers=headers, json=data, timeout=90)
        response.raise_for_status()
        response_json = response.json()
        if not response_json.get("choices"): return [{"Error": "Invalid response structure from DeepSeek API (no choices)."}]
        content = response_json["choices"][0].get("message", {}).get("content", "")
        if not content: return [{"Error": "Empty content in response from DeepSeek API."}]
        # Try parsing as JSON object first, then look for list
        try:
            parsed_json = json.loads(content)
            if isinstance(parsed_json, dict):
                for key, value in parsed_json.items():
                    if isinstance(value, list):
                        return value # Found the list
            # If it's a list directly (less likely with json_object)
            elif isinstance(parsed_json, list):
                return parsed_json
            # If parsing worked but didn't find a list structure
            return [{"Error": "LLM did not return expected JSON list structure (DeepSeek)."}]
        except json.JSONDecodeError:
            # Fallback bracket search if JSON parsing fails
            start_index = content.find('[')
            end_index = content.rfind(']')
            if start_index != -1 and end_index != -1 and start_index < end_index:
                 json_list_str = content[start_index : end_index + 1]
                 try: return json.loads(json_list_str)
                 except json.JSONDecodeError: return [{"Error": "LLM response could not be parsed as JSON list (DeepSeek - Fallback)."}]
            else:
                return [{"Error": "LLM did not return JSON list format (DeepSeek)."}]

    except requests.exceptions.RequestException as e: return [{"Error": f"DeepSeek API call failed (Network/HTTP): {e}"}]
    except Exception as e: return [{"Error": f"Unexpected error during DeepSeek processing: {e}"}]


# --- Central Dispatcher ---
def get_references_from_llm(api_key, llm_provider, text_chunk):
    """Calls the appropriate LLM API based on the provider."""
    if not api_key: return [{"Error": "API Key is missing."}]
    if not text_chunk or not isinstance(text_chunk, str): return [{"Error": "Invalid text chunk provided."}]

    start_time = time.time()
    result = []
    try:
        if llm_provider == "OpenAI":
            result = call_openai_api(api_key, text_chunk)
        elif llm_provider == "Anthropic":
            result = call_anthropic_api(api_key, text_chunk)
        elif llm_provider == "Gemini":
            result = call_gemini_api(api_key, text_chunk)
        elif llm_provider == "DeepSeek":
            result = call_deepseek_api(api_key, text_chunk)
        else:
            result = [{"Error": f"Invalid LLM Provider selected: {llm_provider}"}]
    except Exception as e:
        # Catch errors that might occur *outside* the specific API call functions
        print(f"Critical Error in LLM dispatcher for {llm_provider}: {repr(e)}")
        result = [{"Error": f"Unhandled exception in dispatcher for {llm_provider}. Check logs. Details: {e}"}]
    end_time = time.time()
    print(f"--- LLM Call ({llm_provider}) Duration: {end_time - start_time:.2f} seconds ---")

    # Final check: ensure the result is always a list (even if it's a list containing an error dict)
    if not isinstance(result, list):
        print(f"Error: API function for {llm_provider} returned non-list type: {type(result)}. Wrapping in error list.")
        # Try to include the problematic result in the error message if it's simple
        error_detail = str(result)
        if len(error_detail) > 100:
             error_detail = error_detail[:100] + "..."
        return [{"Error": f"Internal Error: API function for {llm_provider} returned unexpected type ({type(result)}). Content: {error_detail}"}]
    elif len(result) > 0 and not all(isinstance(item, dict) for item in result):
         print(f"Error: API function for {llm_provider} returned list with non-dictionary items: {[type(item) for item in result]}. Wrapping in error list.")
         return [{"Error": f"Internal Error: API function for {llm_provider} returned list with invalid item types."}]

    return result # Should be a list of dicts or a list containing a single error dict

# --- END OF CORRECT FILE chunkrefernce1-main/llm_integrations.py ---
