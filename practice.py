import pymupdf
import requests
import time

URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"

def get_structured_details_from_doc(prompt, messages=None):

    if messages is None:
        messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant, who does exactly as asked to do."
        }
    ]
    
    messages.append({"role": "user", "content": prompt})

    json_data = {
        "model": MODEL,
        "messages": messages,
        "stream": False
    }
    
    response = requests.post(url=URL, json=json_data)
    output = response.json()['message']['content']
    
    messages.append({"role": "assistant", "content": output})
    
    return output, messages
    
    
if __name__ == '__main__':
    
    name = "Kirtan_Parekh_resume.pdf"
    name2 = "test2.pdf"

    # creating reader objs for docs
    doc = pymupdf.open(name)
    doc2 = pymupdf.open(name2)
    
    # get info and remove newlines
    page1 = doc[0]
    page2 = doc2[0]
    
    text1 = page1.get_text().replace("\n", " ")
    text2 = page2.get_text().replace("\n", " ")

    # conversation history for record
    conversation_history = None

    prompt = """
    Below are details of a resume document. However, it's unstructured and jumbled and therefore most of it's details are not in any kind of order. Your task is to use specifically these details and create a more structured format that is easy to embed using embedding model and store in a vector database.
    
    For example, "jennifer lopez i'm a popular singer with millions of fans worldwide. most famous songs are 'get on the floor', 'la la la la', and more." for these toy details, we can make the following structure:
    
    Personal information:
    Name: Jennifer Lopez
    Occupation: famous singer
    
    Experience:
    famous songs are get on the floor and la la la la
    
    Achievements:
    have millions of fans worldwide.
    
    However, this was just a dummy example. Real resumes have a lot of details and the most common sections are: personal details and contact links (example linkedin, gitbub, kaggle, and more), experience, education, projects, skills, certifications. There might be other fields in some particular resumes.
    
    Now, following are the unstructured details of a resume, create a structured one from it and separate it with a newline character. NOTE: Don't generate greeting and help questions at the start of the response or at the end, only return the structured resume details. And DO NOT leave any skill mention in the resume out, must mention it.
    
    details: 
    """
    
    prompt1 = prompt + "['" + text1 + "']"
    prompt2 = prompt + "['" + text2 + "']"
    
    # record time of completion
    start = time.time()
    
    # for resume1
    structured_text1, conversation_history = get_structured_details_from_doc(prompt=prompt1, messages=conversation_history)
    
    # a little delay
    time.sleep(1)
    
    # for resume2
    structured_text2, conversation_history = get_structured_details_from_doc(prompt=prompt2, messages=conversation_history)
    
    end = time.time()
    
    print(structured_text1)
    print(f"\n{"-"*20}\n")
    print(structured_text2)
    print(f"\nTime taken: {round((end-start), 3)} secs.")
    
    with open("history.txt", "w", encoding="utf-8") as file:
        for chat in conversation_history:
            for key, value in chat.items():
                file.write(f"Role :- {str(key)}\n\t\tContent :- {str(value)}\n\n")

