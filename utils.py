import numpy as np
import openai
import pandas as pd
import random
import tiktoken
import time

from vertexai.preview.language_models import TextGenerationModel


QUESTIONS_FILE_PATH = "data/questions_dataset.xlsx"

MAX_RETRIES = 10
INITIAL_DELAY = 1
BACKOFF_FACTOR = 2

MAX_TOKENS = 4096


def tokens_num_allowed(string, encoding_name):
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    tokens_num_allowed = MAX_TOKENS - num_tokens

    return tokens_num_allowed


def prep_dataset(dataframe_path):
    df = pd.read_excel(dataframe_path)
    df["extra_info"].replace(np.nan, None, inplace=True)
    df = df[df["status"] == "OK"]
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    return df


def get_answer(prompt_technique, model, row, temperature=0):
    retries = 0
    df_questions = prep_dataset(QUESTIONS_FILE_PATH)

    while retries < MAX_RETRIES:
        print(f"Retries num {retries}")

        try:

            if prompt_technique == "zero_shot":
                if row["extra_info"] is not None:
                    extra_info_prompt = f"Considere essas informações adicionais:\n\n{row['extra_info']}"
                    prompt = f"Questão: {row['question']}\n\n{extra_info_prompt}\n\nAlternativas:\n\n{row['alternatives']}\n\nResposta: "
                else:
                    prompt = f"Questão: {row['question']}\n\nAlternativas:\n\n{row['alternatives']}\n\nResposta: "

            if prompt_technique == "zero_shot_chain_of_thought":
                zero_shot_cot_prompt = (
                    "Resposta: Vamos pensar passo a passo para resolver a questão."
                )
                if row["extra_info"] is not None:
                    extra_info_prompt = f"Considere essas informações adicionais:\n\n{row['extra_info']}"
                    prompt = f"Questão: {row['question']}\n\n{extra_info_prompt}\n\nAlternativas:\n\n{row['alternatives']}\n\n{zero_shot_cot_prompt}"
                else:
                    prompt = f"Questão: {row['question']}\n\nAlternativas:\n\n{row['alternatives']}\n\n{zero_shot_cot_prompt}"

            if prompt_technique in ["few_shot", "chain_of_thought"]:
                if row["extra_info"] is not None:
                    extra_info_prompt = f"Considere essas informações adicionais:\n\n{row['extra_info']}"
                    prompt = f"Questão: {row['question']}\n\n{extra_info_prompt}\n\nAlternativas:\n\n{row['alternatives']}\n\nResposta: "
                else:
                    prompt = f"Questão: {row['question']}\n\nAlternativas:\n\n{row['alternatives']}\n\nResposta: "

                df_examples = df_questions[
                    (df_questions["exam"] == row["exam"])
                    & (df_questions["year"] != row["year"])
                    & (df_questions["subject"] == row["subject"])
                    & (df_questions[prompt_technique].notnull())
                ]
                prompt_list = []

                for _, row in df_examples.iterrows():
                    if row["extra_info"] is not None:
                        extra_info = f"Considere essas informações adicionais:\n\n{row['extra_info']}"
                        new_prompt = f"Questão: {row['question']}\n\n{extra_info}\n\nAlternativas:\n\n{row['alternatives']}\n\nResposta: {row[prompt_technique]}\n"
                    else:
                        new_prompt = f"Questão: {row['question']}\n\nAlternativas:\n\n{row['alternatives']}\n\nResposta: {row[prompt_technique]}\n"

                    prompt_list.append(new_prompt)

                prompt = "\n".join(prompt_list + [prompt])

            if prompt_technique == "plan_and_solve":
                plan_and_solve_prompt = "Resposta: Vamos primeiro entender o problema, extrair variáveis relevantes, seus números correspondentes e fazer um plano. Então, vamos executar o plano, calcular as variáveis intermediárias (atenção ao cálculo numérico correto e ao bom senso), resolver o problema passo a passo e mostrar a resposta."
                if row["extra_info"] is not None:
                    extra_info_prompt = f"Considere essas informações adicionais:\n\n{row['extra_info']}"
                    prompt = f"Questão: {row['question']}\n\n{extra_info_prompt}\n\nAlternativas:\n\n{row['alternatives']}\n\n{plan_and_solve_prompt}"
                else:
                    prompt = f"Questão: {row['question']}\n\nAlternativas:\n\n{row['alternatives']}\n\n{plan_and_solve_prompt}"

            print(prompt)
            tokens_allowed = tokens_num_allowed(prompt, model)

            if model == "text-davinci-003":
                answer = openai.Completion.create(
                    engine=model,
                    prompt=prompt,
                    temperature=temperature,
                    presence_penalty=0,
                    frequency_penalty=0,
                    max_tokens=tokens_allowed,
                )
                return answer.choices[0].text.strip()

            if model in ["gpt-3.5-turbo-0301", "gpt-4-0314"]:
                answer = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Você é um assistente muito capacitado em resolver questões complexas de Matemática, Química, Física e Português.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    presence_penalty=0,
                    frequency_penalty=0,
                    max_tokens=tokens_allowed,
                )
                return answer.choices[0].message["content"].strip()

            if model == "text-bison@001":
                modelo_instanciado = TextGenerationModel.from_pretrained(model)
                answer = modelo_instanciado.predict(
                    prompt=prompt,
                    temperature=temperature,
                    max_output_tokens=1024,
                    top_k=1,
                    top_p=0.1,
                )
                return answer.text

        except Exception as e:
            if retries == MAX_RETRIES - 1:
                raise e

            delay = INITIAL_DELAY * (BACKOFF_FACTOR ** retries)
            delay += random.randint(0, delay)
            time.sleep(delay)
            retries += 1


def extract_right_option(answer):
    retries = 0

    while retries < MAX_RETRIES:
        try:
            prompt = f"""
            Extraia do texto somente a letra que representa a resposta correta.\n
            Texto: Resposta: d) [18001, 19000].\n
            D\n\n
            Texto: e) 72,8.\n
            E\n\n
            Texto: C) cumprem a função de destacar o absurd\n
            C\n\n
            Texto: {answer}\n
            """
            answer = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0301",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )

            return answer.choices[0].message["content"].strip()

        except Exception as e:
            if retries == MAX_RETRIES - 1:
                raise e

            delay = INITIAL_DELAY * (BACKOFF_FACTOR ** retries)
            delay += random.randint(0, delay)
            time.sleep(delay)
            retries += 1
