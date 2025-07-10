import pytest
import os
from dotenv import load_dotenv
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithoutReference
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompt_values import StringPromptValue
from langchain_core.language_models.llms import LLMResult
from langchain_core.callbacks.manager import Callbacks
from typing import List, Any, Union, Optional
import asyncio
import uuid
import requests
from compatible_chat_openai import CompatibleChatOpenAI

@pytest.mark.asyncio
async def test_context_precision():
    # Load environment variables from .env file
    load_dotenv()
    
    # Use environment variable instead of hardcoded API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Create the LLM instance using our compatible wrapper
    llm = CompatibleChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

    # Create the metric instance
    context_precision = LLMContextPrecisionWithoutReference(llm=llm)
    question = "How many articles are there for JAVA?"

    # Prepare the sample
    # sample = SingleTurnSample(
    #     user_input="How many Cypress Courses are there?",
    #     response="There are 10 Cypress Courses",
    #     retrieved_contexts=["There are 10 Cypress Courses"]
    # )

    # feed data
    response_dict = requests.post("https://rahulshettyacademy.com/rag-llm/ask", json={
        "question": question,
        "chat_history": []
    })

    # Safely extract response data
    response_json = response_dict.json()
    answer = response_json.get("answer", "")
    retrieved_docs = response_json.get("retrieved_docs", [])
    
    # Extract page_content from retrieved docs (safely handle up to 3 docs)
    retrieved_contexts = []
    for i in range(min(3, len(retrieved_docs))):
        if "page_content" in retrieved_docs[i]:
            retrieved_contexts.append(retrieved_docs[i]["page_content"])
    
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=retrieved_contexts
    )

    print("API Response Status:", response_dict.status_code)
    print("API Response Content:", response_dict.text)
    print("API Response JSON:", response_dict.json() if response_dict.status_code == 200 else "No JSON response")
    # Compute the score
    score = await context_precision.single_turn_ascore(sample)
    assert score > 0.8