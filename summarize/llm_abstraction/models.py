import logging

from langchain.chat_models import ChatOpenAI
from langchain.llms import LlamaCpp


def get_model(source: str, kwargs):
    assert "temperature" in kwargs, "temperature must be specified"
    ## Langchain wants temp explicitly stated, so here we go
    temperature = kwargs["temperature"]
    del kwargs["temperature"]
    if source == "chatGPT":
        logging.info("Initializing OpenAI chatGPT LLM")
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo", temperature=temperature, model_kwargs=kwargs
        )
    elif source == "llama-7B":
        logging.info("Initializing Facebook LLaMA 7B model")
        logging.warn("LLaMA is not yet implemented!")
        llm = None
    elif source == "llama-13B":
        logging.info("Initializing Facebook LLaMA 13B model")
        logging.warn("LLaMA is not yet implemented!")
        llm = None
    elif source == "local":
        logging.info("Initializing a locally hosted model ")
        assert "model_path" in kwargs, "model_path must be specified for a local model"
        llm = LlamaCpp(
            model_path=kwargs["model_path"], temperature=temperature, n_ctx=8192
        )
    return llm