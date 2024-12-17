import vertexai.preview.generative_models as generative_models
# from google.oauth2 import service_account
from prompt import system_prompt
import os
from dotenv import load_dotenv
load_dotenv()


generation_config = {
    "max_output_tokens": 1200,
    "temperature": 0.5,
    "top_p": 1,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
}
import google.generativeai as genai
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-pro',generation_config={'temperature':0.3})
prompt=[
        {'role':'user',
        'parts': [system_prompt],
                },
        {'role':'model',
        'parts':["Understood"],
        }]
#history = [{'role': 'user', 'content': [f'{system_prompt}']},
 #           {'role': 'model', 'content': ["Understood"]}]
chat = model.start_chat(history=prompt)


#def multiturn_generate_content(input_message):
    #response = chat.send_message(content=input_message,generation_config=generation_config, safety_settings=safety_settings)
    #return response.text        
async def stream_gemini_response(input_message):
    response = chat.send_message(content=input_message,generation_config=generation_config)
    for chunk in response.text: #Streaming the response , 
        if chunk:
            yield chunk
