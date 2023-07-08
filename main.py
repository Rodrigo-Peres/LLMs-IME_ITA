import argparse
import json
import math
import numpy as np
import openai
import pandas as pd
import time

# from langchain.chat_models import ChatAnthropic, ChatOpenAI  # , ChatVertexAI
# from langchain.llms import OpenAI  # , VertexAI
# from langchain.output_parsers import StructuredOutputParser, ResponseSchema
# from langchain.prompts import (
#    PromptTemplate,
#    ChatPromptTemplate,
#    HumanMessagePromptTemplate,
# )
from utils import get_answer, extract_right_option


with open("./GPT_SECRET_KEY.json") as f:
    open_ai = json.load(f)

openai.api_key = open_ai["API_KEY"]


# model = {
#    "text-davinci-003": OpenAI(
#        model_name="text-davinci-003", temperature=0, openai_api_key=openai.api_key
#    ),
#    "gpt-3.5-turbo": ChatOpenAI(
#        model_name="gpt-3.5-turbo-0301", temperature=0, openai_api_key=openai.api_key
#    ),
#    "gpt-4": ChatOpenAI(
#        model_name="gpt-4-0314", temperature=0, openai_api_key=openai.api_key
#    ),
#    # "text-bison": VertexAI(model_name="text-bison@001", temperature=0),
#    # "chat-bison": ChatVertexAI(model_name="chat-bison@001", temperature=0),
#    # "claude-instant-1": ChatAnthropic(
#    #    model_name="claude-instant-1-100k", temperature=0, openai_api_key=openai.api_key
#    # ),
#    # "claude-1": ChatAnthropic(
#    #    model_name="claude-1-100k", temperature=0, openai_api_key=openai.api_key
#    # ),
# }

models = {
    "davinci": "text-davinci-003",
    "gpt-3.5": "gpt-3.5-turbo-0301",
    "gpt-4": "gpt-4-0314",
    "text-bison": "text-bison@001",
    "chat-bison": "chat-bison@001",
    "claude-instant": "claude-instant-1-100k",
    "claude-1": "claude-1-100k",
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_technique", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    # parser.add_argument("--temperature", type=float, default=None)
    # parser.add_argument("--output_path", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_arguments()
    print("*****************************")
    print(args)
    print("*****************************")

    prompt_technique = args.prompt_technique
    model = models[args.model]

    df = pd.read_excel(r"data/questions_dataset.xlsx", sheet_name="Questions")
    df["extra_info"].replace(np.nan, None, inplace=True)
    print(df.shape)
    df_ok = df[df["status"] == "OK"]
    print(df_ok.shape)
    print(df_ok.head())
    df_ok = df_ok.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    try:
        question_id = df_respostas[df_respostas["model_name"] == model].id.max()
        question_id = 0 if math.isnan(question_id) else question_id
        df_iter = df_ok[df_ok["id"] > question_id]
    except:
        df_respostas = pd.DataFrame(
            columns=["id", "question", "alternatives", "long_answer", "answer"]
        )
        df_iter = df_ok.copy()

    for _, row in df_iter.iterrows():
        print(f"\nQuestion ID {row['id']}")

        long_answer = get_answer(prompt_technique, model, row=row)
        time.sleep(1)
        # answer = extract_right_option(long_answer)
        answer = long_answer

        df_temp = pd.DataFrame(
            {
                "model_name": [model],
                "id": row["id"],
                "question": row["question"],
                "alternatives": row["alternatives"],
                "long_answer": [long_answer],
                "answer": [answer],
            }
        )

        print(f"Retorno: {long_answer}")
        print(f"Resposta final extraída: {answer}\n")

        df_respostas = pd.concat([df_respostas, df_temp], ignore_index=True)
        time.sleep(1)

    # df_respostas.to_excel(rf"data/results/respostas_{model}.xlsx", index=False)


if __name__ == "__main__":
    main()
