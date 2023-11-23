!pip install datasets

from datasets import load_dataset
import pandas as pd

live_qa = dataset = load_dataset("hyesunyun/liveqa_medical_trec2017")
questions = live_qa["test"]["NIST_PARAPHRASE"]

empty_indices = [index for index, q in enumerate(questions) if not q.strip()]
questions[9] = live_qa["test"][9]["NLM_SUMMARY"]
questions[33] = live_qa["test"][33]["NLM_SUMMARY"]
questions[102] = live_qa["test"][102]["NLM_SUMMARY"]

answers = []
for i in live_qa["test"]["REFERENCE_ANSWERS"]:
    answers.append(i[0]["ANSWER"])

liveqa = pd.DataFrame({
    "Question": questions,
    "Answer": answers
})

liveqa.to_csv("path.csv", index=False)
