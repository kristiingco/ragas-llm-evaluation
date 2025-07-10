import pytest
from ragas.metrics import LLMContextRecall

@pytest.mark.asyncio
@pytest.mark.parametrize("get_data", [
   { "question": "How many articles are there for JAVA?", "reference": "23"},
], indirect=True)
async def test_context_recall(llm_wrapper, get_data):
    llm = llm_wrapper
    context_recall = LLMContextRecall(llm=llm)

    sample = get_data  # get_data is already a SingleTurnSample from the fixture

    score = await context_recall.single_turn_ascore(sample)
    print(score)
    assert score > 0.7

