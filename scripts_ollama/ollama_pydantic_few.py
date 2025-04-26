import time
import pandas as pd
from ollama import chat
from pydantic import BaseModel

# Define a Pydantic model for structured response
class Partei(BaseModel):
    name: str
    wahrscheinlichkeit: float

class ParteienResponse(BaseModel):
    parteien: list[Partei]

# Load your dataset with prompts
data = pd.read_csv('filtered_GLES.csv', usecols=['lfdn', 'prompt_A'])  # Replace with your dataset's path

# Define the system prompt
system_prompt = """Aufgabe:
Du bist ein Sozialforscher und auf Basis der folgenden Informationen über eine Person sollst du abschätzen, welche Partei diese Person bei der Bundestagswahl am wahrscheinlichsten wählen würde.
wichtige Hinweise:
Berücksichtige alle aufgeführten Eigenschaften gleichermaßen.
Vermeide dabei einseitige oder klischeehafte Schlussfolgerungen.
Es ist möglich, dass die Person sich nicht in allen Punkten klassisch verhält; bleibe daher vorsichtig in deiner Vorhersage.
Ein gegensätzliches Verhalten ist nicht ausgeschlossen.
Hinterfrage mögliche Stereotype.

Hier auch ein paar Beispiele wie andere Personen gewählt haben:

person A:
Ich bin 31 Jahre alt. Ich bin männlich. Ich habe einen Hochschulabschluss. Ich habe ein mittleres monatliches Haushalts-Nettoeinkommen. Ich bin berufstätig. Ich bin überhaupt nicht religiös. Politisch-ideologisch ordne ich mich mittig links ein. Ich identifiziere mich ziemlich stark mit der Partei Die Linke. Ich lebe in Westdeutschland. Ich finde, die Regierung sollte die Einwanderung weder erleichtern noch einschränken. Ich finde, die Regierung sollte Maßnahmen ergreifen, um die Einkommensunterschiede zu verringern.  -> hat die Linke gewählt

person B:
Ich bin 50 Jahre alt. Ich bin weiblich. Ich habe einen Realschulabschluss. Ich habe ein mittleres monatliches Haushalts-Nettoeinkommen. Ich bin berufstätig. Ich bin etwas religiös. Politisch-ideologisch ordne ich mich in der Mitte ein. Ich identifiziere mich mäßig mit der Partei SPD. Ich lebe in Westdeutschland. Ich finde, die Regierung sollte die Einwanderung erleichtern. Ich finde, die Regierung sollte keine Maßnahmen ergreifen, um die Einkommensunterschiede zu verringern. -> hat die Linke gewählt

person C:
Ich bin 46 Jahre alt. Ich bin männlich. Ich habe Abitur. Ich habe ein hohes monatliches Haushalts-Nettoeinkommen. Ich bin berufstätig. Ich bin überhaupt nicht religiös. Politisch-ideologisch ordne ich mich in der Mitte ein. Ich identifiziere mich ziemlich stark mit der Partei AfD. Ich lebe in Westdeutschland. Ich finde, die Regierung sollte die Einwanderung einschränken. Ich habe keine Meinung dazu, ob die Regierung Maßnahmen ergreifen sollte, um die Einkommensunterschiede zu verringern. -> hat die AfD gewählt

person D:
Ich bin 58 Jahre alt. Ich bin männlich. Ich habe einen Hauptschulabschluss. Ich habe ein mittleres monatliches Haushalts-Nettoeinkommen. Ich bin berufstätig. Ich bin etwas religiös. Politisch-ideologisch ordne ich mich stark links ein. Ich identifiziere mich mäßig mit der Partei AfD. Ich lebe in Westdeutschland. Ich finde, die Regierung sollte die Einwanderung einschränken. Ich finde, die Regierung sollte Maßnahmen ergreifen, um die Einkommensunterschiede zu verringern. -> hat die AfD gewählt

person E:
Ich bin 62 Jahre alt. Ich bin männlich. Ich habe Abitur. Ich habe ein niedriges monatliches Haushalts-Nettoeinkommen. Ich bin nicht berufstätig. Ich bin sehr religiös. Politisch-ideologisch ordne ich mich in der Mitte ein. Ich identifiziere mich mit keiner Partei. Ich lebe in Westdeutschland. Ich finde, die Regierung sollte die Einwanderung erleichtern. Ich finde, die Regierung sollte Maßnahmen ergreifen, um die Einkommensunterschiede zu verringern. -> hat die SPD gewählt

person F:
Ich bin 61 Jahre alt. Ich bin männlich. Ich habe einen Hochschulabschluss. Ich habe ein hohes monatliches Haushalts-Nettoeinkommen. Ich bin berufstätig. Ich bin etwas religiös. Politisch-ideologisch ordne ich mich in der Mitte ein. Ich identifiziere mich ziemlich stark mit der Partei Bündnis 90/Die Grünen. Ich lebe in Westdeutschland. Ich finde, die Regierung sollte die Einwanderung erleichtern. Ich finde, die Regierung sollte Maßnahmen ergreifen, um die Einkommensunterschiede zu verringern. -> hat die Grünen gewählt

Es gibt folgende Parteien zu wählen:
CDU/CSU - SPD - Grünen - AfD - Die Linke - FDP

Antwortformat (nur XML oder JSON, aber es muss kompatibel mit dem definierten Pydantic-Modell sein):

<parteien>
  <partei name="XYZ">Wahrscheinlichkeit</partei>
  <!-- ggf. weitere Parteien -->
</parteien>
(Dabei steht „Wahrscheinlichkeit“ für eine Zahl von 0 bis 100.)
"""

# Create a new dataframe to store the results
results = data.copy()

# Adjust party names to match your schema
parteien = ["CDU/CSU", "SPD", "Die Grünen", "AfD", "Die Linke", "FDP"]

# Initialize probability columns
for partei in parteien:
    results[partei] = 0

# Process each prompt with the LLM
for index, row in data.iterrows():
    # The user's individual info is in 'prompt_A'
    user_prompt = row['prompt_A']
    print(f"Processing index {index}...")

    start_time = time.time()
    
    try:
        # Generate a completion with the ollama 'chat' function
        response = chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            model='llama3.1',
            format=ParteienResponse.model_json_schema(), 
        )

        generation_time = time.time() - start_time
        print(f"Generation completed in {generation_time:.2f} seconds.")

        # Store the raw text in the DataFrame
        results.at[index, "raw_response"] = response.message.content

        # Parse the JSON text inside response.message.content
        parteien_response = ParteienResponse.model_validate_json(response.message.content)

        # Update the columns in 'results'
        for partei_data in parteien_response.parteien:
            partei_name = partei_data.name
            wahrscheinlichkeit = partei_data.wahrscheinlichkeit

            # Safely update only if the column exists
            if partei_name in results.columns:
                results.at[index, partei_name] = wahrscheinlichkeit

    except Exception as e:
        print(f"Error at index {index}: {e}")

# Save the results to a new CSV file
results.to_csv('results_with_predictions.csv', index=False)
