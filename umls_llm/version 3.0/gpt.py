!pip install -q transformers einops accelerate langchain bitsandbytes sentencepiece xformers

from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from torch import cuda
from umls import get_umls_keys
from tqdm import tqdm
import pandas as pd

llm = ChatOpenAI(model_name="MODEL",
                 temperature=0,
                 openai_api_key="KEY")

class ExtendedConversationBufferWindowMemory(ConversationBufferWindowMemory):
    extra_variables:List[str] = []

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables."""
        return [self.memory_key] + self.extra_variables

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return buffer with history and extra variables"""
        d = super().load_memory_variables(inputs)
        d.update({k:inputs.get(k) for k in self.extra_variables})
        return d

memory = ExtendedConversationBufferWindowMemory(k=0,
                                                ai_prefix="Physician",
                                                human_prefix="Patient",
                                                extra_variables=["context"])

template = """
You are an experienced senior physician, please answer the patient's questions in conjunction with the following content.
The content provided may not all be relevant and might not be sufficient to fully answer the question, therefore you need to use your own judgment on how to best respond.
Your answer must conform to medical facts and be as detailed as possible.

Context:
{context}
Current conversation:
{history}
Patient: {input}
Physician:
"""

PROMPT = PromptTemplate(
    input_variables=["context", "history", "input"], template=template
)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=PROMPT,
    verbose=True,
)

# direct extraction
DIRECT_EXTRACTION_PROMPT = """
[INST] Only return the medical terminologies contained in the input question.
Please return in JSON format.

Output Fromat: {"medical terminologies": ["<name>", "<name>"]}

Input: {question}

Output: [/INST]
"""

# indirect extraction
INDIRECT_EXTRACTION_PROMPT = """
[INST] Return medical terminologies related to the input question.
Please return in JSON format.

Output Fromat: {"medical terminologies": ["<name>", "<name>"]}


Input: {question}

Output: [/INST]
"""

questions = pd.read_csv("path.csv")
questions = questions["question_column_name"].tolist() 

results = []
for i in tqdm(questions, desc="Processing Questions"):
    context = get_umls_keys(i)
    answer = conversation.predict(context=context, input=question)
    results.append(answer)

results = pd.DataFrame(results)
results.to_csv("path.csv", index=False)
