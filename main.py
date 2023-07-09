import argparse
import json
import math
import openai
import pandas as pd
import time

from utils import extract_right_option, get_answer, prep_dataset
from models import model_names


QUESTIONS_FILE_PATH = "data/questions_dataset.xlsx"
ANSWERS_FILE_PATH = "data/results/answers.xlsx"

with open("./credentials/GPT_SECRET_KEY.json") as f:
    open_ai = json.load(f)

openai.api_key = open_ai["API_KEY"]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_technique", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0, required=False)

    return parser.parse_args()


def main():
    args = parse_arguments()
    print("*********************")
    print(args)
    print("*********************")

    prompt_technique = args.prompt_technique
    model = model_names[args.model]
    temp = args.temperature

    df_questions = prep_dataset(QUESTIONS_FILE_PATH)

    try:
        df_answers = pd.read_excel(ANSWERS_FILE_PATH)
        question_id = df_answers[
            (df_answers["model_name"] == model)
            & (df_answers["prompt_technique"] == prompt_technique)
        ]["id"].max()
        question_id = 0 if math.isnan(question_id) else question_id
        df_iter = df_questions[df_questions["id"] > question_id]

    except FileNotFoundError:
        df_answers = pd.DataFrame(
            columns=[
                "model_name",
                "prompt_technique",
                "id",
                "question",
                "alternatives",
                "complete_answer",
                "answer",
            ]
        )
        df_iter = df_questions.copy()

    for _, row in df_iter.iterrows():
        print(f"\nQuestion ID {row['id']}")

        complete_answer = get_answer(prompt_technique, model, row=row, temperature=temp)
        print(f"Complete answer: {complete_answer}\n")
        time.sleep(1)
        answer = extract_right_option(complete_answer)
        print(f"Extracted answer: {answer}\n")

        df_temp = pd.DataFrame(
            {
                "model_name": [model],
                "prompt_technique": [prompt_technique],
                "id": row["id"],
                "question": row["question"],
                "alternatives": row["alternatives"],
                "complete_answer": [complete_answer],
                "answer": [answer],
            }
        )

        df_answers = pd.concat([df_answers, df_temp], ignore_index=True)
        time.sleep(1)

    df_answers.to_excel(ANSWERS_FILE_PATH, index=False)


if __name__ == "__main__":
    main()
