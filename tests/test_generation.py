"""
LegalSaarthi — LLM Generation Tests (Phase 12)
"""
import pytest
from src.generation.llm_client import LLMClient
from src.evaluation.generation_metrics import LLMEvaluator

# We can mock the litellm completion to avoid real API calls during unit tests.

class MockResponse:
    def __init__(self, content):
        self.choices = [MockChoice(content)]
        self.usage = {"prompt_tokens": 10, "completion_tokens": 20}
        
    def get(self, key, default):
        if key == "usage":
            return self.usage
        return default

class MockChoice:
    def __init__(self, content):
        self.message = MockMessage(content)

class MockMessage:
    def __init__(self, content):
        self.content = content


def mock_completion(**kwargs):
    messages = kwargs.get("messages", [])
    prompt = messages[-1]["content"] if messages else ""
    
    if "Judge" in kwargs.get("model", ""):
        if "Bad Answer" in prompt:
            return MockResponse("CORRECTNESS: 1\nFACTUALITY: 1")
        return MockResponse("CORRECTNESS: 5\nFACTUALITY: 5")
    
    if "Fail" in prompt:
        raise Exception("Mock failure")
        
    return MockResponse("Mock generated response")


@pytest.fixture
def mock_litellm(monkeypatch):
    monkeypatch.setattr("src.generation.llm_client.completion", mock_completion)


def test_llm_client_generate(mock_litellm):
    client = LLMClient(model_name="test-model")
    res = client.generate("Hello")
    assert res == "Mock generated response"
    assert client.total_prompt_tokens == 10
    assert client.total_completion_tokens == 20


def test_llm_client_retry(mock_litellm):
    client = LLMClient(model_name="test-model")
    # Will fail 3 times and return ERROR
    res = client.generate("Fail", retries=2)
    assert "ERROR" in res


def test_evaluator_abstention(mock_litellm):
    judge = LLMClient(model_name="Judge")
    evaluator = LLMEvaluator(judge)
    
    res = evaluator.evaluate("Q", "Gold", "I don't know the answer.")
    assert res["abstention"] == 1
    assert res["correctness"] == 0
    assert res["factuality"] == 0


def test_evaluator_scoring(mock_litellm):
    judge = LLMClient(model_name="Judge")
    evaluator = LLMEvaluator(judge)
    
    # Should get 5/5
    res = evaluator.evaluate("Q", "Gold", "Good Answer")
    assert res["abstention"] == 0
    assert res["correctness"] == 5
    assert res["factuality"] == 5
    
    # Should get 1/1
    res = evaluator.evaluate("Q", "Gold", "Bad Answer")
    assert res["abstention"] == 0
    assert res["correctness"] == 1
    assert res["factuality"] == 1
