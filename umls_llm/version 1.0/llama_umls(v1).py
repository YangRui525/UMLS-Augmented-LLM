!pip install -q transformers datasets einops accelerate langchain bitsandbytes sentencepiece xformers

import re
import json
import torch
import requests
import transformers
from langchain import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from torch import cuda

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
    token="hf_token"
)

tokenizer = AutoTokenizer.from_pretrained(checkpoint,
                                          token="hf_token")

model.eval()

stop_list = ["\nHuman:", "\n```\n"]
stop_token_ids = [tokenizer(x)["input_ids"] for x in stop_list]
stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
stop_token_ids

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
    stopping_criteria=stopping_criteria,
    repetition_penalty=1.1,
)

llama = HuggingFacePipeline(pipeline=pipeline, model_kwargs = {"temperature":0.0})
llama

memory=ConversationBufferWindowMemory(k=0)

conversation = ConversationChain(
    llm=llama,
    verbose=True,
    memory=memory,
)

B_INST, E_INST = "[INST]", "[/INST]"

prompt = """
<s>[INST] <<SYS>>
The following is a friendly conversation between a patient and an AI doctor. The patient provides additional medical knowledge to the AI doctor. The AI doctor must assess whether this knowledge is useful and integrate useful information into its own knowledge base to better answer patients' questions. The AI doctor only needs to answer questions and does not need to mention anything else. If the AI doctor does not know the answer to a question, it truthfully says it does not know. The AI doctor is talkative and provides lots of specific explanations. The AI doctor answers in a professional tone.
<</SYS>>

Current conversation:
{history}
Human: {input}
AI: [/INST]
"""

conversation.prompt.template = prompt

# direct extraction
EXTRACTION_PROMPT = """
[INST] Only return the medical terminologies contained in the input question.
Please return in JSON format.

Output Fromat: {"medical terminologies": ["<name>", "<name>"]}

Please only return the JSON format information.

Input: {question}

Output: [/INST]"""

# indirect extraction
EXTRACTION_PROMPT = B_INST + """ Return medical terminologies related to the input question. Please return in JSON format.

Output Fromat: {"medical terminologies": ["<name>", "<name>"]}

Please only return the JSON format information.

Input: {question}

Output: """ + E_INST

PROMPT = """{question}

The following medical knowledge may help you:

{context}
Please answer the question with a professional and polite tone, and provide detailed explanations.
"""

class UMLS_API:
	def __init__(self, apikey, version="current"):
		self.apikey = apikey
		self.version = version
		self.search_url = f"https://uts-ws.nlm.nih.gov/rest/search/{version}"
		self.content_url = f"https://uts-ws.nlm.nih.gov/rest/content/{version}"
		self.content_suffix = "/CUI/{}/{}?apiKey={}"

	def search_cui(self, query):
		try:
			page = 1
			size = 1
			query = {"string": query, "apiKey": self.apikey, "pageNumber": page, "pageSize": size}
			r = requests.get(self.search_url, params=query)
			r.raise_for_status()
			print(r.url)
			r.encoding = 'utf-8'
			outputs = r.json()

			items = outputs["result"]["results"]

			if len(items) == 0:
				print("No results found."+"\n")

			cui_results=[]

			for result in items:
				cui_results.append((result["ui"], result["name"]))

		except Exception as except_error:
			print(except_error)

		return cui_results

	def get_definitions(self, cui):
		try:
			suffix = self.content_suffix.format(cui, "definitions", self.apikey)
			r = requests.get(self.content_url + suffix)
			r.raise_for_status()
			r.encoding = "utf-8"
			outputs  = r.json()

			return outputs["result"]
		except Exception as except_error:
			print(except_error)

	def get_relations(self, cui):
		try:
			suffix = self.content_suffix.format(cui, "relations", self.apikey)
			r = requests.get(self.content_url + suffix)
			r.raise_for_status()
			r.encoding = "utf-8"
			outputs  = r.json()

			return outputs["result"]
		except Exception as except_error:
			print(except_error)

umls_api = UMLS_API("API_KEY")

def get_umls_keys(query):
    umls_res = {}

    prompt = EXTRACTION_PROMPT.replace("{question}", query)

    while True:
        try:
            keys_text = llama(prompt)

            print(keys_text)
            pattern = r"\{(.*?)\}"
            matches = re.findall(pattern, keys_text.replace("\n", ""))

            keys_dict = json.loads("{" + matches[0] + "}")

            break
        except Exception as except_error:
            print(except_error)

    for key in keys_dict["medical terminologies"][:]:
        cuis = umls_api.search_cui(key)

        if len(cuis) == 0:
            continue
        cui = cuis[0][0]
        name = cuis[0][1]

        defi = ""
        definitions = umls_api.get_definitions(cui)

        if definitions is not None:
            msh_def = None
            nci_def = None
            icf_def = None
            csp_def = None
            hpo_def = None

            for definition in definitions:
                source = definition["rootSource"]
                if source == "MSH":
                    msh_def = definition["value"]
                    break
                elif source == "NCI":
                    nci_def = definition["value"]
                elif source == "ICF":
                    icf_def = definition["value"]
                elif source == "CSP":
                    csp_def = definition["value"]
                elif source == "HPO":
                    hpo_def = definition["value"]

            defi = msh_def or nci_def or icf_def or csp_def or hpo_def

        relations = umls_api.get_relations(cui)
        rels = []

        if relations is not None:
            for rel in relations[:]:
                related_from_id_name = rel.get("relatedFromIdName", "")
                additional_relation_label = rel.get("additionalRelationLabel", "")
                related_id_name = rel.get("relatedIdName", "")

                if related_from_id_name:
                    rels.append((related_from_id_name, additional_relation_label, related_id_name))

        umls_res[cui] = {"name": name, "definition": defi, "rels": rels}

    context = ""
    for k, v in umls_res.items():
      name = v["name"]
      definition = v["definition"]
      rels = v["rels"]
      rels_text = ""
      for rel in rels:
          rels_text += "(" + rel[0] + "," + rel[1] + "," + rel[2] + ")\n"
      text = f"Name: {name}\nDefinition: {definition}\n"
      if rels_text != "":
          text += f"Relations: {rels_text}"

      context += text + "\n"
    if context != "":
      context = context[:-1]
    return context

# example
question = "A 56 year old male patient with atrial fibrillation presents to the clinic. Given their history of heart failure, diabetes and PAD, what is their risk of stroke? Should they be placed on anticoagulation?"
context = get_umls_keys(question)
prompt = PROMPT.replace("{question}", question).replace("{context}", context)
conversation.predict(input=prompt)
