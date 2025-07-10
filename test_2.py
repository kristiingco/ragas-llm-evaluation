import pytest
from langchain_openai import ChatOpenAI
import requests
import os
from dotenv import load_dotenv
import asyncio
import uuid
from ragas.metrics import LLMContextRecall
from ragas import SingleTurnSample
from compatible_chat_openai import CompatibleChatOpenAI


@pytest.mark.asyncio
async def test_context_recall():
     # Load environment variables from .env file
    load_dotenv()
    
    # Use environment variable instead of hardcoded API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    llm = CompatibleChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    context_recall = LLMContextRecall(llm=llm)

    question = "How many articles are there for JAVA?"


    response_dict = requests.post("https://rahulshettyacademy.com/rag-llm/ask", json={
        "question": question,
        "chat_history": []
    })


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
        reference="23",
        response=answer,
        retrieved_contexts=retrieved_contexts
    )
   
    score = await context_recall.single_turn_ascore(sample)
    print(score)
    assert score > 0.7
