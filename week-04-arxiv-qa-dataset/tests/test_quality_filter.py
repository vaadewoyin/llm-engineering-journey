import pytest
from src.quality_filter import is_valid_chatml_pair

@pytest.fixture
def sample_qa_pairs():
    """Returns a list of Q&A pair dictionaries for testing quality_filter."""
    return [
        {
            "messages": [
                {"role": "user", "content": "This question has enough words to pass the ten word minimum easily."},
                {"role": "assistant", "content": "This answer also has well over twenty words. It provides a detailed explanation that is long enough to meet the threshold requirement for the quality filter."}
            ]
        },  # valid (question >=10 words, answer >=20 words)
        {
            "messages": [
                {"role": "user", "content": "Too short"},
                {"role": "assistant", "content": "This answer is long enough to pass the twenty word minimum because it contains many words here and there and continues to meet the requirement."}
            ]
        },  # invalid: question too short
        {
            "messages": [
                {"role": "user", "content": "This question is long enough to satisfy the ten word threshold easily, yes it is."},
                {"role": "assistant", "content": "Short answer."}
            ]
        },  # invalid: answer too short
    ]

@pytest.fixture
def quality_config():
    """Returns a configuration dictionary for quality filtering."""
    return {
        "min_question_length": 10,
        "min_answer_length": 20,
    }


def test_is_valid_chatml_pair(quality_config, sample_qa_pairs):
    # Expect True for valid chatml pairs
    # The first pair should pass because it meets all length criteria
    assert is_valid_chatml_pair(sample_qa_pairs[0], quality_config)
    # The second pair fails because question too short
    assert not is_valid_chatml_pair(sample_qa_pairs[1], quality_config) 
    # The third fails because answer too short
    assert not is_valid_chatml_pair(sample_qa_pairs[2], quality_config) 
           
