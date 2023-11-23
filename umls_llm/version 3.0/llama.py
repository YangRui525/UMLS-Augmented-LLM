!pip install -q transformers datasets einops accelerate langchain bitsandbytes sentencepiece xformers

import torch
import transformers
from typing import List, Dict, Any
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from torch import cuda
from umls import get_umls_keys

checkpoint = "meta-llama/Llama-2-13b-chat-hf"

device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    trust_remote_code=True,
    quantization_config=nf4_config,
    device_map="auto",
    token="huggingface_TOKEN"
)

tokenizer = AutoTokenizer.from_pretrained(checkpoint,
                                          token="huggingface_TOKEN")

model.eval()

stop_list = ["\nHuman:", "\n```\n"]
stop_token_ids = [tokenizer(x)["input_ids"] for x in stop_list]
stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

pipeline = transformers.pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="cuda",
    max_new_tokens=4096,
    return_full_text=True,
    stopping_criteria=stopping_criteria,
    repetition_penalty=1.1,
)

llama = HuggingFacePipeline(pipeline=pipeline, model_kwargs = {"temperature":0.0})

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
<s>[INST] <<SYS>>
You are an experienced senior physician, please answer the patient's questions in conjunction with the following content.
The content provided may not all be relevant and might not be sufficient to fully answer the question, therefore you need to use your own judgment on how to best respond.
Your answer must conform to medical facts and be as detailed as possible.
<</SYS>>

Context:
{context}
Current conversation:
{history}
Patient: {input}
Physician: [/INST]
"""

PROMPT = PromptTemplate(
    input_variables=["context", "history", "input"], template=template
)

conversation = ConversationChain(
    llm=llama,
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
