import requests
import json
from config import URL, MODEL

def generate_prompt(query, context, has_context):
    """
    Generates a prompt for the LLM based on context availability.
    """
    if has_context:
        prompt = f"""
        Based on the following context from a PDF document, please answer the user's question in a professsional and respectable way.
        Only use information from the context. If the context does not contain the answer, state that you cannot find the answer in the document.
        Also mention the sources of the context at the very end of the response in a professional manner after a few blank lines.

        Context:
        ---
        {context}
        ---

        User Question: {query}
        """
    else:
        prompt = f"""
        The user asked a question, but no relevant information was found in the provided PDF document.
        Please provide a general, helpful answer to the user's question. Do not mention that context was not found.

        User Question: {query}
        """
    return prompt

def get_streamed_response(prompt):
    """
    Gets a streamed response from the local LLM.
    This function specifically looks for <think> tags and yields thought process and answer tokens separately.
    """
    messages = [{"role": "user", "content": prompt}]
    json_data = {"model": MODEL, "messages": messages, "stream": True}
    
    try:
        response = requests.post(url=URL, json=json_data, stream=True)
        response.raise_for_status()
        
        buffer = ""
        in_think_block = False

        for line in response.iter_lines():
            if line:
                try:
                    json_line = json.loads(line)
                    if "message" in json_line and "content" in json_line['message']:
                        token = json_line['message']['content']
                        buffer += token
                        
                        while True:
                            if not in_think_block:
                                start_idx = buffer.find("<think>")
                                if start_idx != -1:
                                    if start_idx > 0:
                                        yield "answer", buffer[:start_idx]
                                    buffer = buffer[start_idx + len("<think>"):]
                                    in_think_block = True
                                else:
                                    yield "answer", buffer
                                    buffer = ""
                                    break
                            else:
                                end_index = buffer.find("</think>")
                                if end_index != -1:
                                    if end_index > 0:
                                        yield "think", buffer[:end_index]
                                    buffer = buffer[end_index + len("</think>"):]
                                    in_think_block = False
                                else:
                                    yield "think", buffer
                                    buffer = ""
                                    break
                except json.JSONDecodeError:
                    continue  # Ignore non-JSON lines
    except requests.RequestException as e:
        print(f"Request failed: {e}")

def stream_handler(prompt):
    """
    A wrapper generator that calls the main streaming function
    and yields only the 'answer' tokens for st.write_stream.
    """
    for part_type, token in get_streamed_response(prompt):
        if part_type == "answer":
            yield token