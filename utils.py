import requests
import os
import json

def load_test_data(file_name):
    test_data_path = os.path.join(os.path.dirname(__file__), "test_data", file_name)
    with open(test_data_path) as file:
        return json.load(file)

def get_llm_response(test_data):
    response_dict = requests.post("https://rahulshettyacademy.com/rag-llm/ask", json={
        "question": test_data["question"],
        "chat_history": []
    })

    return response_dict


   