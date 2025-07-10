import pytest
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
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

class CompatibleChatOpenAI(ChatOpenAI):
    """A wrapper around ChatOpenAI that handles ragas compatibility issues"""
    
    def _convert_to_message_lists(self, messages):
        """Convert various input formats to list[list[BaseMessage]]"""
        if isinstance(messages, StringPromptValue):
            # Convert StringPromptValue to list of BaseMessage lists
            return [[HumanMessage(content=messages.text)]]
        elif isinstance(messages, str):
            # Convert string to list of BaseMessage lists
            return [[HumanMessage(content=messages)]]
        elif isinstance(messages, list):
            if not messages:
                return [[]]
            
            # Check if it's already a list of lists
            if isinstance(messages[0], list):
                # It's already list[list[...]], convert each inner list
                result = []
                for msg_list in messages:
                    converted_list = []
                    for msg in msg_list:
                        if isinstance(msg, str):
                            converted_list.append(HumanMessage(content=msg))
                        elif isinstance(msg, StringPromptValue):
                            converted_list.append(HumanMessage(content=msg.text))
                        elif isinstance(msg, BaseMessage):
                            converted_list.append(msg)
                        else:
                            converted_list.append(HumanMessage(content=str(msg)))
                    result.append(converted_list)
                return result
            else:
                # It's a single list, convert to list of BaseMessages
                converted_list = []
                for msg in messages:
                    if isinstance(msg, str):
                        converted_list.append(HumanMessage(content=msg))
                    elif isinstance(msg, StringPromptValue):
                        converted_list.append(HumanMessage(content=msg.text))
                    elif isinstance(msg, BaseMessage):
                        converted_list.append(msg)
                    else:
                        converted_list.append(HumanMessage(content=str(msg)))
                return [converted_list]
        else:
            # Handle other types by converting to string
            return [[HumanMessage(content=str(messages))]]
    
    async def generate(
        self,
        messages,
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Override generate to handle ragas input formats"""
        # Convert messages to proper format
        converted_messages = self._convert_to_message_lists(messages)
        
        # Call parent agenerate with converted messages (async version)
        return await super().agenerate(
            converted_messages,
            stop=stop,
            callbacks=callbacks,
            **kwargs
        )
    
    async def agenerate(
        self,
        messages,
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Override agenerate to handle ragas input formats"""
        # Convert messages to proper format
        converted_messages = self._convert_to_message_lists(messages)
        
        # Call parent agenerate with converted messages
        return await super().agenerate(
            converted_messages,
            stop=stop,
            callbacks=callbacks,
            **kwargs
        )

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