import pytest
from ragas.metrics import LLMContextRecall

@pytest.mark.asyncio
async def test_context_recall(llm_wrapper, get_data):
    llm = llm_wrapper
    context_recall = LLMContextRecall(llm=llm)

    sample = get_data
   
    score = await context_recall.single_turn_ascore(sample)
    print(score)
    assert score > 0.7

