import pytest 
from ragas.metrics import TopicAdherenceScore
from utils import load_test_data

@pytest.mark.parametrize("get_data", load_test_data("test_4.json"), indirect=True)
@pytest.mark.asyncio 
async def test_topic_adherence(llm_wrapper,get_data):
    topic_score = TopicAdherenceScore(llm=llm_wrapper)
    sample = get_data
    score = await topic_score.single_turn_ascore(sample)
    print(score)
    assert score > 0.8




