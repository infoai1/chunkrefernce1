# --- START OF CORRECT FILE chunkrefernce1-main/llm_integrations.py ---

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

Output Format: Return ONLY a valid JSON list containing one JSON object for each reference found. If no references are found, return an empty list []. Do not include any explanatory text before or after the JSON list.

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
    # Get model from environment variable 'OPENAI_MODEL'.
    default_model = "gpt-4o" # Fallback if OPENAI_MODEL env var is not set
    model_name = os.getenv("OPENAI_MODEL", default_model)

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
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=3000
        )

        content = response.choices[0].message.content

        try:
            parsed_json = json.loads(content)
            if isinstance(parsed_json, dict) and len(parsed_json) == 1:
                 potential_list = list(parsed_json.values())[0]
                 if isinstance(potential_list, list):
                     print(f"--- OpenAI Success: Parsed list from wrapped object. Items: {len(potential_list)} ---")
                     return potential_list
            elif isinstance(parsed_json, list):
                print(f"--- OpenAI Success: Parsed list directly. Items: {len(parsed_json)} ---")
                return parsed_json
            else:
                 print(f"Warning: OpenAI response was valid JSON but not a list or expected wrapper: {content}")
                 return [{"Error": "LLM returned unexpected JSON structure."}]

        except json.JSONDecodeError:
            print(f"Warning: OpenAI response was not valid JSON, attempting bracket search: {content[:200]}...")
            start_index = content.find('[')
            end_index = content.rfind(']')
            if start_index != -1 and end_index != -1 and start_index < end_index:
                 json_list_str = content[start_index : end_index + 1]
                 try:
                     parsed_list = json.loads(json_list_str)
                     print(f"--- OpenAI Success: Parsed list via bracket search. Items: {len(parsed_list)} ---")
                     return parsed_list
                 except json.JSONDecodeError:
                     print(f"Error: Fallback bracket search also failed to parse JSON: {json_list_str[:200]}...")
                     return [{"Error": "LLM response could not be parsed as JSON list."}]
            else:
                print(f"Error: Could not find JSON list brackets '[]' in response: {content[:200]}...")
                return [{"Error": "LLM did not return a JSON list format."}]

    except Exception as e:
        error_message = f"Error calling OpenAI API: {e}"
        print(f"!!! {error_message} !!!")
        if "does not exist" in str(e) or "Invalid model" in str(e) or "model_not_found" in str(e):
            error_detail = f"The specified OpenAI model '{model_name}' was not found or is invalid. Check OPENAI_MODEL environment variable or the default in llm_integrations.py. Original error: {e}"
            print(f"!!! Critical Error: {error_detail} !!!")
            return [{"Error": error_detail}]
        return [{"Error": f"OpenAI API call failed: {e}"}]


# --- Function to call Anthropic ---
def call_anthropic_api(api_key, text_chunk, model="claude-3-opus-20240229"):
    try:
        from anthropic import Anthropic
        print(f"--- Attempting Anthropic API call with model: {model} ---")
        client = Anthropic(api_key=api_key)
        prompt = DETAILED_PROMPT_TEMPLATE.format(text_chunk=text_chunk)
        # ... (rest of Anthropic implementation - kept brief for focus) ...
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
        prompt = DETAILED_PROMPT_TEMPLATE.format(text_chunk=text_chunk)
        # ... (rest of Gemini implementation - kept brief for focus) ...
        generation_config = genai.GenerationConfig(response_mime_type="application/json", temperature=0.2)
        safety_settings=[{"category":c,"threshold":"BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT","HARM_CATEGORY_HATE_SPEECH","HARM_CATEGORY_SEXUALLY_EXPLICIT","HARM_CATEGORY_DANGEROUS_CONTENT"]]
        llm = genai.GenerativeModel(model_name=model, generation_config=generation_config, safety_settings=safety_settings)
        response = llm.generate_content(prompt)
        content = response.text
        try:
             parsed_json = json.loads(content)
             if isinstance(parsed_json, list): return parsed_json
             elif isinstance(parsed_json, dict) and len(parsed_json)==1: # Check wrapped
                  potential_list = list(parsed_json.values())[0]
                  if isinstance(potential_list, list): return potential_list
             return [{"Error": "LLM returned unexpected JSON structure (Gemini)."}]
        except json.JSONDecodeError: return [{"Error": "LLM response could not be parsed as JSON list (Gemini)."}]
        except ValueError as ve: # Safety blocking
             if "response was blocked" in str(ve): return [{"Error": f"Content blocked by Gemini safety filters."}]
             else: raise ve
    except ImportError: return [{"Error": "Google GenerativeAI library not installed."}]
    except Exception as e: return [{"Error": f"Gemini API call failed: {e}"}]


# --- Function to call DeepSeek ---
def call_deepseek_api(api_key, text_chunk, model="deepseek-chat"):
    try:
        API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions")
        print(f"--- Attempting DeepSeek API call with model: {model} ---")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        prompt_content = DETAILED_PROMPT_TEMPLATE.format(text_chunk=text_chunk)
        # ... (rest of DeepSeek implementation - kept brief for focus) ...
        data = {"model": model, "messages": [{"role":"system", "content":"Respond ONLY with valid JSON list."},{"role":"user","content":prompt_content}], "max_tokens":3500, "temperature":0.2}
        response = requests.post(API_URL, headers=headers, json=data, timeout=90)
        response.raise_for_status()
        response_json = response.json()
        if not response_json.get("choices"): return [{"Error": "Invalid response structure from DeepSeek API (no choices)."}]
        content = response_json["choices"][0].get("message", {}).get("content", "")
        if not content: return [{"Error": "Empty content in response from DeepSeek API."}]
        start_index = content.find('[')
        end_index = content.rfind(']')
        if start_index != -1 and end_index != -1 and start_index < end_index:
             json_list_str = content[start_index : end_index + 1]
             try: return json.loads(json_list_str)
             except json.JSONDecodeError: return [{"Error": "LLM response could not be parsed as JSON list (DeepSeek)."}]
        else: # Fallback check direct parse
            try:
                parsed_list = json.loads(content)
                if isinstance(parsed_list, list): return parsed_list
            except json.JSONDecodeError: pass
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
        print(f"Critical Error in LLM dispatcher for {llm_provider}: {repr(e)}")
        result = [{"Error": f"Unhandled exception in dispatcher for {llm_provider}. Check logs."}]
    end_time = time.time()
    print(f"--- LLM Call ({llm_provider}) Duration: {end_time - start_time:.2f} seconds ---")
    if not isinstance(result, list):
        print(f"Error: API function for {llm_provider} returned non-list type: {type(result)}. Wrapping.")
        return [{"Error": f"Internal Error: API function for {llm_provider} returned unexpected type."}]
    return result

# --- END OF CORRECT FILE chunkrefernce1-main/llm_integrations.py ---
