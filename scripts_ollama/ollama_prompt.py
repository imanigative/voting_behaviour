import time
import pandas as pd
from ollama import chat
from pydantic import BaseModel


# Define a Pydantic model for structured response
class Partei(BaseModel):
    Partei: str
    wahrscheinlichkeit: float

class ParteienResponse(BaseModel):
    parteien: list[Partei]

# Load your dataset with prompts
data = pd.read_csv('GLES2017_A.csv', usecols=['lfdn', 'prompt_A'])  # Replace with your dataset's path

# Define the system prompt
system_prompt = """Aufgabe:
Du bist ein Sozialforscher und auf Basis der folgenden Informationen über eine Person sollst du abschätzen, welche Partei diese Person bei der Bundestagswahl am wahrscheinlichsten wählen würde.
wichtige Hinweise:
Berücksichtige alle aufgeführten Eigenschaften gleichermaßen.
Vermeide dabei einseitige oder klischeehafte Schlussfolgerungen.
Es ist möglich, dass die Person sich nicht in allen Punkten klassisch verhält; bleibe daher vorsichtig in deiner Vorhersage.
Ein gegensätzliches Verhalten ist nicht ausgeschlossen.
Hinterfrage mögliche Stereotype.
"""

system_prompt = """
Es gibt folgende Parteien zu wählen:
Union - SPD - Grünen - Afd - Die Linke - FDP
Antwortformat (nur XML):
<parteien>
  <partei name=\"XYZ\">Wahrscheinlichkeit</partei>
  <!-- ggf. weitere Parteien -->
</parteien>
(Dabei steht „Wahrscheinlichkeit“ für eine Zahl von 0 bis 100.)"""

# Create a new dataframe to store the results
results = data.copy()
parteien = ["CDU", "SPD", "Grüne", "FDP", "AfD", "Linke"]
for partei in parteien:
    results[partei] = 0  # Initialize probability columns

# Process each prompt with the LLM
for index, row in data.iterrows():
    prompt = row['prompt_A']
    full_prompt = f"{system_prompt}\n\n{prompt}"

    print(f"Processing index {index}: {prompt}")
    start_time = time.time()

    try:
        # Generate a completion
       response = chat(
     messages=[
    {
      'role': 'user',
      'content': 'Tell me about Canada.',
    }
  ],
  model='llama3.1',
  format=Country.model_json_schema(),
)

        # Measure the generation time
        generation_time = time.time() - start_time
        print(f"Generation completed in {generation_time:.2f} seconds.")


        for partei_data in parteien_response.parteien:
            partei_name = partei_data.name
            wahrscheinlichkeit = partei_data.wahrscheinlichkeit

            # Update the corresponding column in results
            if partei_name in results.columns:
                results.at[index, partei_name] = wahrscheinlichkeit



# Save the results to a new CSV file
results.to_csv('results_with_predictions.csv', index=False)

