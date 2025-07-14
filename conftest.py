import pytest
import requests
import os
from dotenv import load_dotenv
from ragas import SingleTurnSample, MultiTurnSample, HumanMessage, AIMessage
from compatible_chat_openai import CompatibleChatOpenAI
from utils import get_llm_response

# Load environment variables
load_dotenv()

@pytest.fixture 
def llm_wrapper():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    llm = CompatibleChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    return llm

@pytest.fixture
def get_data(request):
    test_data = request.param

    response_dict = get_llm_response(test_data)

    response_json = response_dict.json()
    answer = response_json.get("answer", "")
    retrieved_docs = response_json.get("retrieved_docs", [])
    
    # Extract page_content from retrieved docs (safely handle up to 3 docs)
    retrieved_contexts = []
    for i in range(min(3, len(retrieved_docs))):
        if "page_content" in retrieved_docs[i]:
            retrieved_contexts.append(retrieved_docs[i]["page_content"])

    sample = SingleTurnSample(
        user_input=test_data["question"],
        reference=test_data["reference"],
        response=answer,
        retrieved_contexts=retrieved_contexts
    )

    return sample

@pytest.fixture
def get_data_conversation(request):
    test_data = request.param

    response_dict = get_llm_response(test_data)

    response_json = response_dict.json()
    answer = response_json.get("answer", "")
    retrieved_docs = response_json.get("retrieved_docs", [])

    conversation = [
        HumanMessage(content="How many courses are there?"),
        AIMessage(content="There are 10 courses."),
        HumanMessage(content=test_data["question"]),
        AIMessage(content=answer)
    ]

    reference_topics = [
        "courses",
        "students",
        "JavaScript",
        "Cypress"
    ]
    # Extract page_content from retrieved docs (safely handle up to 3 docs)
    retrieved_contexts = []
    for i in range(min(3, len(retrieved_docs))):
        if "page_content" in retrieved_docs[i]:
            retrieved_contexts.append(retrieved_docs[i]["page_content"])



    sample = MultiTurnSample(
        user_input=conversation,
        reference_topics=reference_topics,
    )

    return sample
    