import time
import pandas as pd
import xml.etree.ElementTree as ET
from typing import List, Optional
from pydantic import BaseModel
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser

# -----------------------------------------------------------------------------
# 1. Pydantic models for structured responses
# -----------------------------------------------------------------------------
class Partei(BaseModel):
    name: str
    wahrscheinlichkeit: float

class ParteienResponse(BaseModel):
    parteien: List[Partei]

# -----------------------------------------------------------------------------
# 2. Custom output parser to extract the XML into Pydantic objects
# -----------------------------------------------------------------------------
class ParteienParser(BaseOutputParser):
    def parse(self, text: str) -> ParteienResponse:
        """
        Expects an XML structure like:
          <parteien>
            <partei name="XYZ">50</partei>
            ...
          </parteien>
        Returns a ParteienResponse object.
        """
        try:
            root = ET.fromstring(text.strip())
            parteien_list = []
            for partei_el in root.findall('partei'):
                name = partei_el.attrib.get('name', '')
                # Convert probability text to float (e.g. "50" -> 50.0)
                try:
                    wahrscheinlichkeit = float(partei_el.text.strip())
                except ValueError:
                    wahrscheinlichkeit = 0.0
                parteien_list.append(Partei(name=name, wahrscheinlichkeit=wahrscheinlichkeit))
            return ParteienResponse(parteien=parteien_list)
        except ET.ParseError as e:
            # If parsing fails, return an empty structure or raise an error
            raise ValueError(f"XML parsing error: {e}")

# -----------------------------------------------------------------------------
# 3. Create a LangChain LLM instance (Ollama)
#    - Adjust host, model, or other parameters as needed
# -----------------------------------------------------------------------------
llm = Ollama(
    base_url="http://127.0.0.1:11434",  # if your Ollama server is at localhost:11434
    model="llama3.1",                   # adjust to whichever model you want
)

# -----------------------------------------------------------------------------
# 4. Define the system prompt (and incorporate user prompt at runtime)
# -----------------------------------------------------------------------------
system_prompt = (
    "Du bist ein Sozialforscher und sollst auf Basis der folgenden Informationen "
    "über eine Person abschätzen, welche Partei diese Person wahrscheinlich wählen würde.\n"
    "Vermeide einseitige oder klischeehafte Schlüsse. Hinterfrage mögliche Stereotype.\n\n"
    "Parteien: Union - SPD - Grünen - Afd - Die Linke - FDP\n"
    "Erwarte reines XML als Antwort:\n"
    "<parteien>\n"
    "  <partei name=\"XYZ\">[0-100]</partei>\n"
    "</parteien>"
)

# You can wrap this in a PromptTemplate if you wish to pass a variable prompt
prompt_template = PromptTemplate(
    input_variables=["person_data"],
    template="""
{system_prompt}

Informationen:
{person_data}

Antwort bitte in reinem XML-Format.
""".strip()
)

# -----------------------------------------------------------------------------
# 5. Create an LLMChain with your prompt template and a custom parser
# -----------------------------------------------------------------------------
chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    output_parser=ParteienParser()
)

# -----------------------------------------------------------------------------
# 6. Load your dataset
# -----------------------------------------------------------------------------
data = pd.read_csv('GLES2017_A.csv', usecols=['lfdn', 'prompt_A'])  # adjust path if needed

# Create a DataFrame for results, and initialize columns for each party
results = data.copy()
parteien = ["CDU", "SPD", "Grüne", "FDP", "AfD", "Linke"]
for partei in parteien:
    results[partei] = 0.0  # initialize as float

# -----------------------------------------------------------------------------
# 7. Process each row in the dataset
# -----------------------------------------------------------------------------
for index, row in data.iterrows():
    prompt_content = row['prompt_A']

    print(f"\nProcessing index {index}: {prompt_content}")

    start_time = time.time()

    # Run the chain with the person's info
    try:
        response_obj: ParteienResponse = chain.run(
            person_data=prompt_content,
            system_prompt=system_prompt
        )

        generation_time = time.time() - start_time
        print(f"Generation completed in {generation_time:.2f} seconds.")
        
        # Update DataFrame with extracted probabilities
        for partei_data in response_obj.parteien:
            if partei_data.name in results.columns:
                results.at[index, partei_data.name] = partei_data.wahrscheinlichkeit

    except Exception as e:
        print(f"Error processing index {index}: {e}")

    # User prompt to continue or break (optional)
    user_input = input("Continue processing? (yes to continue, no to stop): ").strip().lower()
    if user_input == 'no':
        break

# -----------------------------------------------------------------------------
# 8. Save the results to a new CSV
# -----------------------------------------------------------------------------
results.to_csv('results_with_predictions.csv', index=False)
print("Results saved to 'results_with_predictions.csv'.")
