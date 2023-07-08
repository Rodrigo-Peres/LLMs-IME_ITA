import numpy as np
import openai
import pandas as pd
import random
import time

from vertexai.preview.language_models import TextGenerationModel


MAX_RETRIES = 10
INITIAL_DELAY = 1
BACKOFF_FACTOR = 2


prompt_input = "Responda a questão e selecione a alternativa correta."  ###


def get_answer(prompt_technique, model, row, temperature=0):
    retries = 0  # Contador de tentativas

    df = pd.read_excel(r"data/questions_dataset.xlsx", sheet_name="Questions")
    df["extra_info"].replace(np.nan, None, inplace=True)
    df_ok = df[df["status"] == "OK"]
    df_ok = df_ok.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    while retries < MAX_RETRIES:
        print(f"retries num {retries}")

        try:

            if prompt_technique == "zero_shot":
                if row["extra_info"] is not None:
                    extra_info_prompt = f"Considere essas informações adicionais:\n\n{row['extra_info']}"
                    prompt = f"{prompt_input}\n\nPergunta: {row['question']}\n\n{extra_info_prompt}\n\nAlternativas:\n\n{row['alternatives']}"
                else:
                    prompt = f"{prompt_input}\n\nPergunta: {row['question']}\n\nAlternativas:\n\n{row['alternatives']}"

            if prompt_technique == "zero_shot_chain_of_thought":
                zero_shot_cot_prompt = "Pense passo a passo para responder a questão."
                if row["extra_info"] is not None:
                    extra_info_prompt = f"Considere essas informações adicionais:\n\n{row['extra_info']}"
                    prompt = f"{prompt_input}\n\nPergunta: {row['question']}\n\n{extra_info_prompt}\n\nAlternativas:\n\n{row['alternatives']}\n\n{zero_shot_cot_prompt}"
                else:
                    prompt = f"{prompt_input}\n\nPergunta: {row['question']}\n\nAlternativas:\n\n{row['alternatives']}\n\n{zero_shot_cot_prompt}"

            if prompt_technique in ["few_shot", "chain_of_thought"]:
                if row["extra_info"] is not None:
                    extra_info_prompt = f"Considere essas informações adicionais:\n\n{row['extra_info']}"
                    prompt = f"{prompt_input}\n\nPergunta: {row['question']}\n\n{extra_info_prompt}\n\nAlternativas:\n{row['alternatives']}"  ###
                else:
                    prompt = f"{prompt_input}\n\nPergunta: {row['question']}\n\nAlternativas:\n\n{row['alternatives']}"  ###

                df_examples = df_ok[
                    (df_ok["exam"] == row["exam"])
                    & (df_ok["year"] != row["year"])
                    & (df_ok["subject"] == row["subject"])
                    & (df_ok[prompt_technique].notnull())
                ]
                few_shot_prompt = []

                for _, row in df_examples.iterrows():
                    if row["extra_info"] is not None:
                        extra_info = f"Considere essas informações adicionais:\n\n{row['extra_info']}"
                        new_prompt = f"Pergunta: {row['question']}\n\n{extra_info}\n\nAlternativas:\n\n{row['alternatives']}\n\n{row[prompt_technique]}\n"
                    else:
                        new_prompt = f"Pergunta: {row['question']}\n\nAlternativas:\n\n{row['alternatives']}\n\n{row[prompt_technique]}\n"

                    few_shot_prompt.append(new_prompt)

                prompt = "\n".join(few_shot_prompt + [prompt])

            if prompt_technique == "plan_and_solve":
                plan_and_solve_prompt = "Vamos primeiro entender o problema, extrair variáveis relevantes e seus numerais correspondentes e fazer um plano. Então, vamos executar o plano, calcular as variáveis intermediárias (atenção ao cálculo numérico correto e ao bom senso), resolver o problema passo a passo e mostrar a resposta."
                if row["extra_info"] is not None:
                    extra_info_prompt = f"Considere essas informações adicionais:\n\n{row['extra_info']}"
                    prompt = f"{prompt_input}\n\nPergunta: {row['question']}\n\n{extra_info_prompt}\n\nAlternativas:\n\n{row['alternatives']}\n\n{plan_and_solve_prompt}"
                else:
                    prompt = f"{prompt_input}\n\nPergunta: {row['question']}\n\nAlternativas:\n\n{row['alternatives']}\n\n{plan_and_solve_prompt}"

            print(prompt)
            return prompt

        #            if model == "text-davinci-003":
        #                resposta = openai.Completion.create(
        #                    engine=model, prompt=prompt, temperature=temperature
        #                )
        #                return resposta.choices[0].text.strip()
        #
        #            if model in ["gpt-3.5-turbo-0301", "gpt-4-0314"]:
        #                resposta = openai.ChatCompletion.create(
        #                    model=model,
        #                    messages=[
        #                        {
        #                            "role": "system",
        #                            "content": "Você é um assistente muito capacitado em resolver questões complexas de Matemática, Química, Física e Português.",
        #                        },
        #                        {"role": "user", "content": prompt},
        #                    ],
        #                    temperature=temperature,
        #                )
        #                return resposta.choices[0].message["content"].strip()
        #
        #            if model == "text-bison@001":
        #                modelo_instanciado = TextGenerationModel.from_pretrained(model)
        #                resposta = modelo_instanciado.predict(
        #                    prompt=prompt,
        #                    temperature=temperature,
        #                    max_output_tokens=1024,
        #                    top_k=1,
        #                    top_p=0.1,
        #                )
        #                return resposta.text

        except Exception as e:
            if retries == MAX_RETRIES - 1:
                raise e

            delay = INITIAL_DELAY * (BACKOFF_FACTOR ** retries)
            delay += random.randint(0, delay)
            time.sleep(delay)
            retries += 1


def extract_right_option(resposta):
    retries = 0  # Contador de tentativas

    while retries < MAX_RETRIES:
        try:
            prompt = f"Extraia somente a letra da resposta escolhida como correta texto a seguir.\nEssa letra geralmente vai ter o seguinte formato 'a)' e seu retorno deverá vir em maiúsculo, dessa forma: 'A'\nTexto: {resposta}\n"
            resposta = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0301",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )

            return resposta.choices[0].message["content"].strip()

        except Exception as e:
            if retries == MAX_RETRIES - 1:
                raise e

            delay = INITIAL_DELAY * (BACKOFF_FACTOR ** retries)
            delay += random.randint(0, delay)
            time.sleep(delay)
            retries += 1
