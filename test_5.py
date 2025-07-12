import pytest
import asyncio
from ragas import evaluate, EvaluationDataset
from ragas.metrics import ResponseRelevancy, FactualCorrectness
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings
from utils import load_test_data

@pytest.mark.parametrize("get_data", load_test_data("test_5.json"), indirect=True)
@pytest.mark.asyncio
async def test_relevancy_factual(llm_wrapper,get_data):
    # Create embeddings instance using langchain's OpenAIEmbeddings
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    
    # Create metrics with embeddings for ResponseRelevancy
    metrics = [ResponseRelevancy(llm=llm_wrapper, embeddings=embeddings), FactualCorrectness(llm=llm_wrapper)]

    dataset = EvaluationDataset([get_data])
    results = evaluate(dataset=dataset, metrics=metrics)
    
    # Extract the first (and only) score from each result list
    response_relevancy_score = results['answer_relevancy'][0]
    factual_correctness_score = results['factual_correctness(mode=f1)'][0]
    
    print(f"Response Relevancy Score: {response_relevancy_score}")
    print(f"Factual Correctness Score: {factual_correctness_score}")
    
    # Assert that both scores are above 0.8
    assert response_relevancy_score > 0.8, f"Response relevancy score {response_relevancy_score} is not above 0.8"
    assert factual_correctness_score > 0.8, f"Factual correctness score {factual_correctness_score} is not above 0.8"