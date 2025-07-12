import pytest
import asyncio
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
    sample = get_data
    scores = await asyncio.gather(*[metric.single_turn_ascore(sample) for metric in metrics])
    print(scores)
    assert scores[0] > 0.8 and scores[1] > 0.8