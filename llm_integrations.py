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

Output Format: Return ONLY a valid JSON list containing one JSON object for each reference found. If no references are found, return an empty list []. Do not include any explanatory text, code formatting (like ```json), or markdown before or after the JSON list. Just the raw list.

Text Chunk to Analyze:
---
{text_chunk}
---

JSON Output:
""" # Reverted prompt instruction to ask for a plain list

# --- Function to call OpenAI ---
def call_openai_api(api_key, text_chunk): # No model default here
    if OpenAI is None:
        return [{"Error": "OpenAI library not installed. Please install it."}]

    # --- OpenAI GPT Configuration ---
    # Explicitly set the desired model. Using gpt-4-turbo.
    # User mentioned "gpt-4.1" - ensure 'gpt-4-turbo' corresponds to the intended model.
    model_name = "gpt-4.1"

    print(f"--- Attempting OpenAI API call with model: {model_name} ---")

    try:
        client = OpenAI(api_key=api_key)
        prompt = DETAILED_PROMPT_TEMPLATE.format(text_chunk=text_chunk)

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert reference extractor outputting ONLY a valid JSON list as described."},
                {"role": "user", "content": prompt}
            ],
            # response_format={"type": "json_object"}, # REMOVED this parameter again
            temperature=0.2,
            max_tokens=3000
        )

        content = response.choices[0].message.content.strip() # Strip leading/trailing whitespace

        # --- RESTORED PARSING LOGIC (for non-guaranteed JSON) ---
        try:
            # Attempt 1: Direct JSON list parsing
            parsed_json = json.loads(content)
            if isinstance(parsed_json, list):
                # Check if items in the list are dictionaries
                if all(isinstance(item, dict) for item in parsed_json):
                    print(f"--- OpenAI Success: Parsed list directly. Items: {len(parsed_json)} ---")
                    return parsed_json
                elif len(parsed_json) == 0: # Handle empty list case
                     print(f"--- OpenAI Success: Parsed empty list directly. ---")
                     return parsed_json
                else:
                    print(f"Warning: OpenAI response was a list but contained non-dict items: {content}")
                    return [{"Error": "LLM returned list with non-dictionary items."}]
            # Handle if it returns a dict wrapping the list
            elif isinstance(parsed_json, dict):
                 found_list = None
                 for key, value in parsed_json.items():
                     if isinstance(value, list):
                         # Check if items in the found list are dictionaries
                         if all(isinstance(item, dict) for item in value):
                             found_list = value
                             print(f"--- OpenAI Success: Parsed list from wrapper dict key '{key}'. Items: {len(found_list)} ---")
                             break
                         elif len(value) == 0: # Handle empty list in wrapper
                              found_list = value
                              print(f"--- OpenAI Success: Parsed empty list from wrapper dict key '{key}'. ---")
                              break
                 if found_list is not None:
                     return found_list
                 else:
                     print(f"Warning: OpenAI response was JSON object but no valid list found: {content}")
                     return [{"Error": "LLM returned unexpected JSON object structure (no valid list)."}]
            else:
                # Parsed as JSON, but not a list or dict?
                print(f"Warning: OpenAI response was valid JSON but not a list or expected wrapper: {content}")
                return [{"Error": "LLM returned unexpected JSON structure (not list/dict)."}]

        except json.JSONDecodeError:
            # Attempt 2: Response might contain markdown/text around the JSON list. Find brackets.
            print(f"Warning: OpenAI response was not valid JSON directly, attempting bracket search: {content[:200]}...")
            start_index = content.find('[')
            end_index = content.rfind(']')
            # Ensure brackets enclose potentially valid JSON list content
            if start_index != -1 and end_index != -1 and start_index < end_index:
                 # Check if there's non-whitespace text BEFORE the opening bracket
                 if content[:start_index].strip():
                      print(f"Warning: Text found before opening bracket: '{content[:start_index].strip()}'")
                 # Check if there's non-whitespace text AFTER the closing bracket
                 if content[end_index + 1:].strip():
                      print(f"Warning: Text found after closing bracket: '{content[end_index + 1:].strip()}'")

                 json_list_str = content[start_index : end_index + 1]
                 try:
                     parsed_list = json.loads(json_list_str)
                     # Check if the parsed result is a list and contains dictionaries
                     if isinstance(parsed_list, list):
                         if all(isinstance(item, dict) for item in parsed_list) or len(parsed_list) == 0:
                             print(f"--- OpenAI Success: Parsed list via bracket search. Items: {len(parsed_list)} ---")
                             return parsed_list
                         else:
                             print(f"Warning: Bracket search found list with non-dict items: {json_list_str}")
                             return [{"Error": "LLM response bracket search found list with non-dictionary items."}]
                     else:
                        print(f"Error: Bracket search parsed JSON, but it wasn't a list: {type(parsed_list)}")
                        return [{"Error": "LLM response bracket search parsed non-list JSON."}]
                 except json.JSONDecodeError:
                     # Bracket search found brackets, but content inside wasn't valid JSON list
                     print(f"Error: Fallback bracket search failed to parse JSON: {json_list_str[:200]}...")
                     return [{"Error": "LLM response contained brackets but invalid JSON list inside."}]
            else:
                # Could not find list brackets at all
                print(f"Error: Could not find JSON list brackets '[]' in response: {content[:200]}...")
                return [{"Error": "LLM did not return text containing a JSON list format."}]
        # --- END RESTORED PARSING LOGIC ---

    except Exception as e:
        # General catch-all for API call issues (network, auth, rate limits, etc.)
        # or unexpected errors during response processing.
        error_message = f"Error calling OpenAI API or processing response: {e}"
        print(f"!!! {error_message} !!!")
        # Keep the specific model error check
        if "does not exist" in str(e) or "Invalid model" in str(e) or "model_not_found" in str(e):
            error_detail = f"The specified OpenAI model '{model_name}' was not found or is invalid. Check the model name in llm_integrations.py. Original error: {e}"
            print(f"!!! Critical Error: {error_detail} !!!")
            return [{"Error": error_detail}]
        # Include the type of exception for better debugging if it's not a model error
        return [{"Error": f"OpenAI API call failed ({type(e).__name__}): {e}"}]


# --- Function to call Anthropic ---
# (Keep Anthropic, Gemini, DeepSeek functions as they were - no changes needed there)
def call_anthropic_api(api_key, text_chunk, model="claude-3-opus-20240229"):
    try:
        from anthropic import Anthropic
        print(f"--- Attempting Anthropic API call with model: {model} ---")
        client = Anthropic(api_key=api_key)
        prompt = DETAILED_PROMPT_TEMPLATE.format(text_chunk=text_chunk) # Use original prompt asking for list
        message = client.messages.create(
             model=model, max_tokens=3500, system="Respond ONLY with valid JSON list.",
             messages=[{"role": "user", "content": prompt}]
             )
        content = message.content[0].text.strip()
        start_index = content.find('[')
        end_index = content.rfind(']')
        if start_index != -1 and end_index != -1 and start_index < end_index:
             json_list_str = content[start_index : end_index + 1]
             try:
                 parsed_list = json.loads(json_list_str)
                 if isinstance(parsed_list, list):
                      return parsed_list
                 else:
                      return [{"Error": "LLM response bracket search parsed non-list JSON (Anthropic)."}]
             except json.JSONDecodeError: return [{"Error": "LLM response could not be parsed as JSON list (Anthropic)."}]
        else: return [{"Error": "LLM did not return a JSON list format (Anthropic)."}]
    except ImportError: return [{"Error": "Anthropic library not installed."}]
    except Exception as e: return [{"Error": f"Anthropic API call failed ({type(e).__name__}): {e}"}]


# --- Function to call Gemini ---
def call_gemini_api(api_key, text_chunk, model="gemini-1.5-pro-latest"):
    try:
        import google.generativeai as genai
        print(f"--- Attempting Gemini API call with model: {model} ---")
        genai.configure(api_key=api_key)
        prompt = DETAILED_PROMPT_TEMPLATE.format(text_chunk=text_chunk) # Use original prompt asking for list
        generation_config = genai.GenerationConfig(response_mime_type="application/json", temperature=0.2)
        safety_settings=[{"category":c,"threshold":"BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT","HARM_CATEGORY_HATE_SPEECH","HARM_CATEGORY_SEXUALLY_EXPLICIT","HARM_CATEGORY_DANGEROUS_CONTENT"]]
        llm = genai.GenerativeModel(model_name=model, generation_config=generation_config, safety_settings=safety_settings)
        response = llm.generate_content(prompt)
        # Gemini with response_mime_type='application/json' should return parsable JSON string in response.text
        content = response.text.strip()
        try:
             parsed_json = json.loads(content)
             if isinstance(parsed_json, list): return parsed_json
             elif isinstance(parsed_json, dict): # Check wrapped object
                  for key, value in parsed_json.items():
                      if isinstance(value, list):
                          return value
             return [{"Error": "LLM returned unexpected JSON structure (Gemini)."}]
        except json.JSONDecodeError:
             # Fallback bracket search just in case Gemini adds text despite mime type
            start_index = content.find('[')
            end_index = content.rfind(']')
            if start_index != -1 and end_index != -1 and start_index < end_index:
                 json_list_str = content[start_index : end_index + 1]
                 try:
                     parsed_list = json.loads(json_list_str)
                     if isinstance(parsed_list, list): return parsed_list
                 except json.JSONDecodeError: pass # Fall through if bracket search fails
            return [{"Error": "LLM response could not be parsed as JSON list (Gemini)."}]
        except ValueError as ve: # Safety blocking
             if "response was blocked" in str(ve): return [{"Error": f"Content blocked by Gemini safety filters."}]
             else: raise ve # Re-raise other ValueErrors
    except ImportError: return [{"Error": "Google GenerativeAI library not installed."}]
    except Exception as e: return [{"Error": f"Gemini API call failed ({type(e).__name__}): {e}"}]


# --- Function to call DeepSeek ---
def call_deepseek_api(api_key, text_chunk, model="deepseek-chat"):
    try:
        API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions")
        print(f"--- Attempting DeepSeek API call with model: {model} ---")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        prompt_content = DETAILED_PROMPT_TEMPLATE.format(text_chunk=text_chunk) # Use original prompt asking for list
        data = {"model": model, "messages": [{"role":"system", "content":"Respond ONLY with valid JSON list."},{"role":"user","content":prompt_content}], "max_tokens":3500, "temperature":0.2, "response_format": {"type": "json_object"}}
        response = requests.post(API_URL, headers=headers, json=data, timeout=90)
        response.raise_for_status()
        response_json = response.json()
        if not response_json.get("choices"): return [{"Error": "Invalid response structure from DeepSeek API (no choices)."}]
        content = response_json["choices"][0].get("message", {}).get("content", "").strip()
        if not content: return [{"Error": "Empty content in response from DeepSeek API."}]
        # Try parsing as JSON object first, then look for list
        try:
            parsed_json = json.loads(content)
            if isinstance(parsed_json, dict):
                for key, value in parsed_json.items():
                    if isinstance(value, list):
                        return value # Found the list
            elif isinstance(parsed_json, list):
                return parsed_json
            return [{"Error": "LLM did not return expected JSON list structure (DeepSeek)."}]
        except json.JSONDecodeError:
            # Fallback bracket search if JSON parsing fails
            start_index = content.find('[')
            end_index = content.rfind(']')
            if start_index != -1 and end_index != -1 and start_index < end_index:
                 json_list_str = content[start_index : end_index + 1]
                 try:
                     parsed_list = json.loads(json_list_str)
                     if isinstance(parsed_list, list): return parsed_list
                 except json.JSONDecodeError: pass # Fall through
            return [{"Error": "LLM response could not be parsed as JSON list (DeepSeek - Fallback)."}]

    except requests.exceptions.RequestException as e: return [{"Error": f"DeepSeek API call failed (Network/HTTP): {e}"}]
    except Exception as e: return [{"Error": f"Unexpected error during DeepSeek processing ({type(e).__name__}): {e}"}]


# --- Central Dispatcher ---
# (Keep dispatcher logic as is)
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
        print(f"Critical Error in LLM dispatcher for {llm_provider}: {repr(e)}")
        result = [{"Error": f"Unhandled exception in dispatcher for {llm_provider}. Check logs. Details: {e}"}]
    end_time = time.time()
    print(f"--- LLM Call ({llm_provider}) Duration: {end_time - start_time:.2f} seconds ---")

    # Final check: ensure the result is always a list of dicts or a list containing a single error dict
    if not isinstance(result, list):
        print(f"Error: API function for {llm_provider} returned non-list type: {type(result)}. Wrapping in error list.")
        error_detail = str(result)
        if len(error_detail) > 100:
             error_detail = error_detail[:100] + "..."
        return [{"Error": f"Internal Error: API function for {llm_provider} returned unexpected type ({type(result)}). Content: {error_detail}"}]
    # Ensure all items in a non-empty list are dictionaries
    elif len(result) > 0 and not all(isinstance(item, dict) for item in result):
         print(f"Error: API function for {llm_provider} returned list with non-dictionary items: {[type(item) for item in result]}. Wrapping in error list.")
         # Try to show the problematic items if simple
         error_detail = str(result)
         if len(error_detail) > 150:
              error_detail = error_detail[:150] + "..."
         return [{"Error": f"Internal Error: API function for {llm_provider} returned list with invalid item types. Content: {error_detail}"}]

    return result

# --- END OF CORRECT FILE chunkrefernce1-main/llm_integrations.py ---
