from llmgpt.config import get_config


def test_config_acts_as_a_dict_in_retrieval():
    cfg = get_config()
    assert isinstance(cfg["model_weights_file"], str)
    assert isinstance(cfg["gpt_config"]["context_length"], int)
