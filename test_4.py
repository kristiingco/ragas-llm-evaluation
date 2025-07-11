import pytest 
from ragas.metrics import Faithfulness
from utils import load_test_data

@pytest.mark.parametrize("get_data", load_test_data("test_4.json"), indirect=True)
@pytest.mark.asyncio 
async def test_faithfulness(llm_wrapper,get_data):
    faithful = Faithfulness(llm=llm_wrapper)
    sample = get_data
    score = await faithful.single_turn_ascore(sample)
    print(score)
    assert score > 0.8

