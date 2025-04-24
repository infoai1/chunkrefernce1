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
# **** MODIFIED PROMPT ****
DETAILED_PROMPT_TEMPLATE = """
You are an expert reference extractor. Your primary goal is to meticulously analyze the text chunk provided below and extract ALL meaningful references according to the detailed schema described. Be thorough and identify every reference present, even if implicit or relating to broader concepts mentioned.

For EACH reference found, return a JSON object with the following keys:
"Reference Category", "Reference Domain", "Reference Detail", "Tags", "Reference Source Name",
"Author Mentioned", "Is Religious?", "Direct Quote", "New Term/Concept?",
"Date or Time Reference", "Place or Location", "Language of Source", "Summary of Relevance".

SCHEMA DETAILS: (Examples added/modified for clarity)
- Reference Category: Specific type (e.g., Quran Verse, Hadith, Historical Event, Scientific Fact, Poetry Line, Dream, New Term by Author, Quote from Person (even unnamed like 'Western philosopher'), Philosophical Concept, etc.)
- Reference Domain: Broad category (Religious & Scriptural, Historical & Biographical, Literary & Cultural, Academic & Intellectual, Personal & Experiential, Scientific & Technical, Social, Economic & Political)
- Reference Detail: Precise description (e.g., "Quran 2:255 (Ayat al-Kursi)", "Hadith on Patience from Sahih Bukhari", "1947 Partition of India", "Concept of 'God's Creation Plan' as defined by author", "Quote from unnamed Western philosopher on man's place in the universe")
- Tags: 3-5 concise keywords (e.g., ["Quran", "Tawhid", "Ayat al-Kursi"], ["Hadith", "Patience", "Bukhari"], ["Philosophy", "Existentialism", "Quote"], ["Creation Plan", "Theology", "Purpose"])
- Reference Source Name: Specific text/book/scholar/person cited (e.g., "Sahih Bukhari", "Quran", "Author's observation", "Ihya Ulum al-Din", "Unnamed Western philosopher")
- Author Mentioned: Name of person mentioned, if any (e.g., "Prophet Muhammad", "Ghazali", "None", "Western philosopher")
- Is Religious?: Boolean true/false.
- Direct Quote: The exact quoted text, if present (e.g., "'The strong is not the one who overcomes...'"). Return empty string "" if none.
- New Term/Concept?: The specific term/concept if unique to the author or central to the argument (e.g., "God's Creation Plan", "Divine character"). Return empty string "" if none.
- Date or Time Reference: Extracted date/year/era (e.g., "7th century", "1988"). Return "None" as a string if none.
- Place or Location: Extracted place (e.g., "Mecca", "India"). Return "None" as a string if none.
- Language of Source: Original language if known (e.g., "Arabic", "Urdu", "English"). Return "Unknown" if not determinable.
- Summary of Relevance: 1-2 sentences on how the reference is used in the chunk.

Output Format: Return ONLY a valid JSON list containing one JSON object for each reference found. If absolutely no references meeting the schema are found, return an empty list []. Do not include any explanatory text, code formatting (like ```json), or markdown before or after the JSON list. Just the raw JSON list itself.

Text Chunk to Analyze:
---
{text_chunk}
---

JSON Output:
"""
# **** END MODIFIED PROMPT ****


# --- Function to call OpenAI ---
def call_openai_api(api_key, text_chunk):
    if OpenAI is None:
        return [{"Error": "OpenAI library not installed. Please install it."}]

    # Explicitly set the desired model. Using gpt-4-turbo.
    # Ensure 'gpt-4-turbo' is the correct identifier for the model you intend.
    model_name = "gpt-4-turbo" # Use the official model name

    print(f"--- Attempting OpenAI API call with model: {model_name} ---")

    try:
        client = OpenAI(api_key=api_key)
        # Use the modified prompt
        prompt = DETAILED_PROMPT_TEMPLATE.format(text_chunk=text_chunk)

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert reference extractor. Your goal is to find all references in the text according to the provided schema and output them ONLY as a valid JSON list."}, # System prompt reinforces goal and format
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, # Low temp for consistency
            max_tokens=3500 # Increased slightly just in case more references are found
        )

        content = response.choices[0].message.content.strip() # Strip leading/trailing whitespace
        # **** ADDED PRINT STATEMENT ****
        print(f"--- RAW OpenAI Response for chunk starting '{text_chunk[:50]}...':\n{content}\n--- END RAW ---")
        # **** END ADDED PRINT STATEMENT ****

        # --- Parsing Logic (Keep the robust version) ---
        try:
            # Attempt 1: Direct JSON list parsing
            parsed_json = json.loads(content)
            if isinstance(parsed_json, list):
                if all(isinstance(item, dict) for item in parsed_json) or len(parsed_json) == 0:
                    print(f"--- OpenAI Success: Parsed list directly. Items: {len(parsed_json)} ---")
                    return parsed_json
                else:
                    print(f"Warning: OpenAI response was a list but contained non-dict items: {content}")
                    return [{"Error": "LLM returned list with non-dictionary items."}]
            # Handle if it returns a dict wrapping the list
            elif isinstance(parsed_json, dict):
                 found_list = None
                 for key, value in parsed_json.items():
                     if isinstance(value, list):
                         if all(isinstance(item, dict) for item in value) or len(value) == 0:
                             found_list = value
                             print(f"--- OpenAI Success: Parsed list from wrapper dict key '{key}'. Items: {len(found_list)} ---")
                             break
                 if found_list is not None:
                     return found_list
                 else:
                     print(f"Warning: OpenAI response was JSON object but no valid list found: {content}")
                     return [{"Error": "LLM returned unexpected JSON object structure (no valid list)."}]
            else:
                print(f"Warning: OpenAI response was valid JSON but not a list or expected wrapper: {content}")
                return [{"Error": "LLM returned unexpected JSON structure (not list/dict)."}]

        except json.JSONDecodeError:
            # Attempt 2: Bracket search fallback
            print(f"Warning: OpenAI response was not valid JSON directly, attempting bracket search: {content[:200]}...")
            start_index = content.find('[')
            end_index = content.rfind(']')
            if start_index != -1 and end_index != -1 and start_index < end_index:
                 if content[:start_index].strip():
                      print(f"Warning: Text found before opening bracket: '{content[:start_index].strip()}'")
                 if content[end_index + 1:].strip():
                      print(f"Warning: Text found after closing bracket: '{content[end_index + 1:].strip()}'")
                 json_list_str = content[start_index : end_index + 1]
                 try:
                     parsed_list = json.loads(json_list_str)
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
                     print(f"Error: Fallback bracket search failed to parse JSON: {json_list_str[:200]}...")
                     return [{"Error": "LLM response contained brackets but invalid JSON list inside."}]
            else:
                print(f"Error: Could not find JSON list brackets '[]' in response: {content[:200]}...")
                return [{"Error": "LLM did not return text containing a JSON list format."}]
        # --- End Parsing Logic ---

    except Exception as e:
        error_message = f"Error calling OpenAI API or processing response: {e}"
        print(f"!!! {error_message} !!!")
        if "does not exist" in str(e) or "Invalid model" in str(e) or "model_not_found" in str(e):
            error_detail = f"The specified OpenAI model '{model_name}' was not found or is invalid. Check the model name in llm_integrations.py. Original error: {e}"
            print(f"!!! Critical Error: {error_detail} !!!")
            return [{"Error": error_detail}]
        return [{"Error": f"OpenAI API call failed ({type(e).__name__}): {e}"}]


# --- Function to call Anthropic ---
def call_anthropic_api(api_key, text_chunk, model="claude-3-opus-20240229"):
    try:
        from anthropic import Anthropic
        print(f"--- Attempting Anthropic API call with model: {model} ---")
        client = Anthropic(api_key=api_key)
        prompt = DETAILED_PROMPT_TEMPLATE.format(text_chunk=text_chunk) # Use updated prompt
        message = client.messages.create(
             model=model, max_tokens=3500, system="Respond ONLY with valid JSON list.",
             messages=[{"role": "user", "content": prompt}]
             )
        content = message.content[0].text.strip()
        # **** ADD THIS PRINT STATEMENT ****
        print(f"--- RAW Anthropic Response for chunk starting '{text_chunk[:50]}...':\n{content}\n--- END RAW ---")
        # **** END ADDED PRINT STATEMENT ****
        start_index = content.find('[')
        end_index = content.rfind(']')
        if start_index != -1 and end_index != -1 and start_index < end_index:
             json_list_str = content[start_index : end_index + 1]
             try:
                 parsed_list = json.loads(json_list_str)
                 if isinstance(parsed_list, list):
                      # Further check if list items are dicts
                      if all(isinstance(item, dict) for item in parsed_list) or len(parsed_list) == 0:
                           print(f"--- Anthropic Success: Parsed list via bracket search. Items: {len(parsed_list)} ---")
                           return parsed_list
                      else:
                           print(f"Warning: Anthropic bracket search found list with non-dict items: {json_list_str}")
                           return [{"Error": "LLM response bracket search found list with non-dictionary items (Anthropic)."}]
                 else:
                      print(f"Error: Anthropic bracket search parsed non-list JSON: {type(parsed_list)}")
                      return [{"Error": "LLM response bracket search parsed non-list JSON (Anthropic)."}]
             except json.JSONDecodeError:
                 print(f"Error: Anthropic bracket search failed to parse JSON: {json_list_str[:200]}...")
                 return [{"Error": "LLM response could not be parsed as JSON list (Anthropic)."}]
        else:
             print(f"Error: Could not find JSON list brackets '[]' in Anthropic response: {content[:200]}...")
             return [{"Error": "LLM did not return a JSON list format (Anthropic)."}]
    except ImportError: return [{"Error": "Anthropic library not installed."}]
    except Exception as e:
         print(f"!!! Error calling Anthropic API: {e} !!!")
         return [{"Error": f"Anthropic API call failed ({type(e).__name__}): {e}"}]


# --- Function to call Gemini ---
def call_gemini_api(api_key, text_chunk, model="gemini-1.5-pro-latest"):
    try:
        import google.generativeai as genai
        print(f"--- Attempting Gemini API call with model: {model} ---")
        genai.configure(api_key=api_key)
        prompt = DETAILED_PROMPT_TEMPLATE.format(text_chunk=text_chunk) # Use updated prompt
        generation_config = genai.GenerationConfig(response_mime_type="application/json", temperature=0.2)
        safety_settings=[{"category":c,"threshold":"BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT","HARM_CATEGORY_HATE_SPEECH","HARM_CATEGORY_SEXUALLY_EXPLICIT","HARM_CATEGORY_DANGEROUS_CONTENT"]]
        llm = genai.GenerativeModel(model_name=model, generation_config=generation_config, safety_settings=safety_settings)
        response = llm.generate_content(prompt)
        content = response.text.strip()
        # **** ADD THIS PRINT STATEMENT ****
        print(f"--- RAW Gemini Response for chunk starting '{text_chunk[:50]}...':\n{content}\n--- END RAW ---")
        # **** END ADDED PRINT STATEMENT ****
        try:
             parsed_json = json.loads(content)
             if isinstance(parsed_json, list):
                 if all(isinstance(item, dict) for item in parsed_json) or len(parsed_json) == 0:
                     print(f"--- Gemini Success: Parsed list directly. Items: {len(parsed_json)} ---")
                     return parsed_json
                 else:
                     print(f"Warning: Gemini response was list with non-dict items: {content}")
                     return [{"Error": "LLM returned list with non-dictionary items (Gemini)."}]
             elif isinstance(parsed_json, dict): # Check wrapped object
                  found_list = None
                  for key, value in parsed_json.items():
                      if isinstance(value, list):
                          if all(isinstance(item, dict) for item in value) or len(value) == 0:
                              found_list = value
                              print(f"--- Gemini Success: Parsed list from wrapper dict key '{key}'. Items: {len(found_list)} ---")
                              break
                  if found_list is not None:
                      return found_list
                  else:
                      print(f"Warning: Gemini response was JSON object but no valid list found: {content}")
                      return [{"Error": "LLM returned unexpected JSON object structure (Gemini)."}]
             else:
                print(f"Warning: Gemini response was valid JSON but not list/dict: {content}")
                return [{"Error": "LLM returned unexpected JSON structure (not list/dict) (Gemini)."}]
        except json.JSONDecodeError:
            # Fallback bracket search just in case Gemini adds text despite mime type
            print(f"Warning: Gemini response was not valid JSON directly, attempting bracket search: {content[:200]}...")
            start_index = content.find('[')
            end_index = content.rfind(']')
            if start_index != -1 and end_index != -1 and start_index < end_index:
                 json_list_str = content[start_index : end_index + 1]
                 try:
                     parsed_list = json.loads(json_list_str)
                     if isinstance(parsed_list, list):
                         if all(isinstance(item, dict) for item in parsed_list) or len(parsed_list) == 0:
                              print(f"--- Gemini Success: Parsed list via bracket search. Items: {len(parsed_list)} ---")
                              return parsed_list
                         else:
                              print(f"Warning: Gemini bracket search found list with non-dict items: {json_list_str}")
                              return [{"Error": "LLM response bracket search found list with non-dictionary items (Gemini)."}]
                     else:
                          print(f"Error: Gemini bracket search parsed non-list JSON: {type(parsed_list)}")
                          return [{"Error": "LLM response bracket search parsed non-list JSON (Gemini)."}]
                 except json.JSONDecodeError:
                     print(f"Error: Gemini fallback bracket search failed to parse JSON: {json_list_str[:200]}...")
                     pass # Fall through to the final error
            print(f"Error: Could not parse Gemini response as JSON list: {content[:200]}...")
            return [{"Error": "LLM response could not be parsed as JSON list (Gemini)."}]
        except ValueError as ve: # Safety blocking
             if "response was blocked" in str(ve):
                 print(f"Warning: Gemini content blocked by safety filters.")
                 return [{"Error": f"Content blocked by Gemini safety filters."}]
             else:
                 print(f"!!! Value Error during Gemini processing: {ve} !!!")
                 raise ve # Re-raise other ValueErrors
    except ImportError: return [{"Error": "Google GenerativeAI library not installed."}]
    except Exception as e:
         print(f"!!! Error calling Gemini API: {e} !!!")
         return [{"Error": f"Gemini API call failed ({type(e).__name__}): {e}"}]


# --- Function to call DeepSeek ---
def call_deepseek_api(api_key, text_chunk, model="deepseek-chat"):
    try:
        API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions")
        print(f"--- Attempting DeepSeek API call with model: {model} ---")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        prompt_content = DETAILED_PROMPT_TEMPLATE.format(text_chunk=text_chunk) # Use updated prompt
        data = {"model": model, "messages": [{"role":"system", "content":"Respond ONLY with valid JSON list."},{"role":"user","content":prompt_content}], "max_tokens":3500, "temperature":0.2}
        # Add response_format if supported (check DeepSeek docs for model compatibility)
        # data["response_format"] = {"type": "json_object"}

        response = requests.post(API_URL, headers=headers, json=data, timeout=90)
        response.raise_for_status()
        response_json = response.json()
        if not response_json.get("choices"): return [{"Error": "Invalid response structure from DeepSeek API (no choices)."}]
        content = response_json["choices"][0].get("message", {}).get("content", "").strip()
        if not content: return [{"Error": "Empty content in response from DeepSeek API."}]
        # **** ADD THIS PRINT STATEMENT ****
        print(f"--- RAW DeepSeek Response for chunk starting '{text_chunk[:50]}...':\n{content}\n--- END RAW ---")
        # **** END ADDED PRINT STATEMENT ****
        # Try parsing as list first (primary instruction)
        try:
            parsed_json = json.loads(content)
            if isinstance(parsed_json, list):
                if all(isinstance(item, dict) for item in parsed_json) or len(parsed_json) == 0:
                    print(f"--- DeepSeek Success: Parsed list directly. Items: {len(parsed_json)} ---")
                    return parsed_json
                else:
                    print(f"Warning: DeepSeek response was list with non-dict items: {content}")
                    return [{"Error": "LLM returned list with non-dictionary items (DeepSeek)."}]
            # Fallback: check if it wrapped it in a dict
            elif isinstance(parsed_json, dict):
                found_list = None
                for key, value in parsed_json.items():
                    if isinstance(value, list):
                        if all(isinstance(item, dict) for item in value) or len(value) == 0:
                            found_list = value
                            print(f"--- DeepSeek Success: Parsed list from wrapper dict key '{key}'. Items: {len(found_list)} ---")
                            break
                if found_list is not None:
                    return found_list
                else:
                    print(f"Warning: DeepSeek response was JSON object but no valid list found: {content}")
                    return [{"Error": "LLM returned unexpected JSON object structure (DeepSeek)."}]
            else:
                 print(f"Warning: DeepSeek response was valid JSON but not list/dict: {content}")
                 return [{"Error": "LLM returned unexpected JSON structure (not list/dict) (DeepSeek)."}]

        except json.JSONDecodeError:
            # Fallback bracket search if JSON parsing fails
            print(f"Warning: DeepSeek response was not valid JSON directly, attempting bracket search: {content[:200]}...")
            start_index = content.find('[')
            end_index = content.rfind(']')
            if start_index != -1 and end_index != -1 and start_index < end_index:
                 json_list_str = content[start_index : end_index + 1]
                 try:
                     parsed_list = json.loads(json_list_str)
                     if isinstance(parsed_list, list):
                         if all(isinstance(item, dict) for item in parsed_list) or len(parsed_list) == 0:
                              print(f"--- DeepSeek Success: Parsed list via bracket search. Items: {len(parsed_list)} ---")
                              return parsed_list
                         else:
                              print(f"Warning: DeepSeek bracket search found list with non-dict items: {json_list_str}")
                              return [{"Error": "LLM response bracket search found list with non-dictionary items (DeepSeek)."}]
                     else:
                          print(f"Error: DeepSeek bracket search parsed non-list JSON: {type(parsed_list)}")
                          return [{"Error": "LLM response bracket search parsed non-list JSON (DeepSeek)."}]
                 except json.JSONDecodeError:
                     print(f"Error: DeepSeek fallback bracket search failed to parse JSON: {json_list_str[:200]}...")
                     pass # Fall through
            print(f"Error: Could not parse DeepSeek response as JSON list: {content[:200]}...")
            return [{"Error": "LLM response could not be parsed as JSON list (DeepSeek - Fallback)."}]

    except requests.exceptions.RequestException as e:
         print(f"!!! DeepSeek API call failed (Network/HTTP): {e} !!!")
         return [{"Error": f"DeepSeek API call failed (Network/HTTP): {e}"}]
    except Exception as e:
         print(f"!!! Unexpected error during DeepSeek processing: {e} !!!")
         return [{"Error": f"Unexpected error during DeepSeek processing ({type(e).__name__}): {e}"}]


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
    elif len(result) > 0 and not all(isinstance(item, dict) for item in result):
         print(f"Error: API function for {llm_provider} returned list with non-dictionary items: {[type(item) for item in result]}. Wrapping in error list.")
         error_detail = str(result)
         if len(error_detail) > 150:
              error_detail = error_detail[:150] + "..."
         return [{"Error": f"Internal Error: API function for {llm_provider} returned list with invalid item types. Content: {error_detail}"}]

    return result

# --- END OF FILE chunkrefernce1-main/llm_integrations.py ---
