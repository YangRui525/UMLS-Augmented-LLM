!pip install -q datasets openai langchain transformers

import re
import json
import torch
import requests
import transformers
from typing import List, Dict, Any

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

    prompt = INDIRECT_EXTRACTION_PROMPT.replace("{question}", query)

    while True:
        try:
            keys_text = llm.predict(prompt)

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
question = "What are the references with noonan syndrome and polycystic renal disease?"
context = get_umls_keys(question)
conversation.predict(context=context, input=question)
