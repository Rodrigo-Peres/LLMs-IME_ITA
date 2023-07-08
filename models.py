import openai

from langchain.llms import OpenAI, VertexAI
from langchain.chat_models import ChatAnthropic, ChatOpenAI, ChatVertexAI


model = {
    "text-davinci-003": OpenAI(
        model_name="text-davinci-003", temperature=0, openai_api_key=openai.api_key
    ),
    "gpt-3.5-turbo": ChatOpenAI(
        model_name="gpt-3.5-turbo-0301", temperature=0, openai_api_key=openai.api_key
    ),
    "gpt-4": ChatOpenAI(
        model_name="gpt-4-0314", temperature=0, openai_api_key=openai.api_key
    ),
    "text-bison": VertexAI(model_name="text-bison@001", temperature=0),
    "chat-bison": ChatVertexAI(model_name="chat-bison@001", temperature=0),
    "claude-instant-1": ChatAnthropic(
        model_name="claude-instant-1-100k", temperature=0, openai_api_key=openai.api_key
    ),
    "claude-1": ChatAnthropic(
        model_name="claude-1-100k", temperature=0, openai_api_key=openai.api_key
    ),
}
