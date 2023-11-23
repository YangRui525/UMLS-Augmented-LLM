!pip install rouge evaluate bert_score
!git clone https://github.com/andersjo/pyrouge.git

import pandas as pd
from rouge import Rouge
from evaluate import load

data = pd.read_csv("path.csv")
reference_answers = data["answer_column_name"].tolist() 

results = pd.read_csv("path.csv")

rouge = Rouge()
rouge_scores = rouge.get_scores(reference_answers, results, avg=True)
print(rouge_scores)

bertscore = load("bertscore")
bert_score = bertscore.compute(predictions=results, references=reference_answers, lang="en", verbose=True)
print(sum(rouge_scores["f1"])/len(reference_answers), 
      sum(rouge_scores["precision"])/reference_answers, 
      sum(rouge_scores["recall"])/len(reference_answers))
