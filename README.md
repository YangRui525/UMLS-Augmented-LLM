# UMLS-Augmented-LLM

Large language models (LLMs) have brought unprecedented innovation to the healthcare field. Despite the promising application in healthcare, they face significant challenges since the content they generate might deviate from established medical facts and even exhibit potential biases. We develop an augmented LLM framework based on the Unified Medical Language System (UMLS), aiming to better serve the healthcare community. It's noteworthy that multiple resident physicians conducted blind reviews of the generated content, adn the results indicate that our framework effectively enhances the factuality, completeness, and relevance of the generated content.

### Framework for Augmenting LLMs with UMLS Database




## Table of Contents

* [Contributors](#contributors)
* [Contents](#contents)
* [Data](#data)
* [Notebooks](#notebooks)
* [Dependencies](#dependencies)


## Contributors
Rui Yang, Edison Marrese-Taylor, Yuhe Ke, Lechao Cheng, Qingyu Chen, Irene Li

<!-- Contents -->
## Contents
The volume of research literature on critical care has experienced exponential growth over the past two decades, rendering traditional bibliometric methods inadequate for analyzing such vast datasets. This necessitates the employment of machine learning (ML) and natural language processing (NLP) techniques to investigate research trends and uncover knowledge gaps. In this study, we utilized uniform manifold approximation and projection (UMAP) and the BERTopic algorithm to explore the differences and similarities between open-access database studies and traditional intensive care studies. Our objective is to identify existing knowledge gaps and explore ways in which they can complement each other.<br />

<!-- Data -->
## Data
Publication abstracts were obtained from Web of Science. The search result was on January 18, 2023. <br />

<!-- Notebooks -->
## Notebooks
* [UMAP](UMAP.ipynb): jupyter notebook with UMAP algorithm
* [BERTopic](BERTopic.ipynb): jupyter notebook with BERTopic algorithm

<!-- Citation -->
## Citation
```bibtext
@misc{yang2023umlsaugmented,
      title={A UMLS-Augmented Framework for Improving Factuality in Large Language Models within Healthcare}, 
      author={Rui Yang and Edison Marrese-Taylor and Yuhe Ke and Lechao Cheng and Qingyu Chen and Irene Li},
      year={2023},
      eprint={2310.02778},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

