import os
import openai
import pandas as pd

# -----------------------------------------------------------------------------
# 1) OPENAI KEY SETUP
# -----------------------------------------------------------------------------
openai.api_key = "IHRE_OPENAI_API_KEY"  # Bitte Ihren Key einsetzen

# -----------------------------------------------------------------------------
# 2) LADEN DES DATENSATZES
# -----------------------------------------------------------------------------
df = pd.read_csv("GLES2017.csv")  
# Beispiel: df hat Spalten wie "age", "female", "edu", "leftright", "partyid", etc.

# -----------------------------------------------------------------------------
# 3) FUNKTION ZUR PERSONA-GENERIERUNG
# -----------------------------------------------------------------------------
def generate_persona(row: pd.Series) -> str:
    """
    Erstellt auf Basis der Merkmale in 'row' einen Persona-Text
    (z. B. Name, Alter, politische Orientierung, ...).
    """
    # Relevante Felder aus dem Datensatz
    age = row.get("age", "unbekanntes Alter")
    female_flag = row.get("female", 0)
    gender = "weiblich" if female_flag == 1 else "männlich"
    edu = row.get("edu", "unbekannter Bildungshintergrund")
    emp = row.get("emp", "unbekannt")
    income = row.get("hhincome", "unbekannt")
    eastwest = row.get("east", "unbekannt")
    religion = row.get("religious", "unbekannt")
    left_right = row.get("leftright", "unbekannte Ausprägung")
    partyid = row.get("partyid", "unbekannte Parteizugehörigkeit")
    partyid_degree = row.get("partyid_degree", "unbekannte Parteizugehörigkeit")
    imigration = row.get("imigration", "unbekannt")
    inequality = row.get("inequality", "unbekannt")

    # Prompt-Text für ChatCompletion
    persona_prompt = f"""
    Erstelle eine kurze, fiktive Persona aus folgenden Daten:
    - Alter: {age}
    - Geschlecht: {gender}
    - Bildung: {edu}
    - Beschäftigungsstatus: {emp}
    - Einkommen: {income}
    - Region: {eastwest}
    - Religiosität: {religion}
    - Links-Rechts-Skala: {left_right}
    - Parteizugehörigkeit: {partyid}
    - Grad der Parteizugehörigkeit: {partyid_degree}
    - Einstellung zur Einwanderung: {imigration}
    - Einstellung zur Ungleichheit: {inequality}

    Schreibe 1-2 Sätze, die diese Person beschreiben.
    """

    # # LLM-Abfrage
    # try:
    #     response = openai.ChatCompletion.create(
    #         model="gpt-3.5-turbo",  # oder GPT-4, falls verfügbar
    #         messages=[
    #             {"role": "user", "content": persona_prompt}
    #         ],
    #         max_tokens=100,
    #         temperature=0.7
    #     )
    #     persona_text = response.choices[0].message.content.strip()
    # except Exception as e:
    #     print(f"Fehler bei Persona-Generierung: {e}")
    #     persona_text = "Persona konnte nicht generiert werden."
    
    return persona_prompt


generate_persona(df.iloc[0])
# -----------------------------------------------------------------------------
# 4) FUNKTION ZUR VORHERSAGE DES WAHLVERHALTENS
# -----------------------------------------------------------------------------
def predict_vote(persona_text: str) -> str:
    """
    Nimmt eine generierte Persona als Input und bittet das LLM,
    eine Wahlvorhersage zu treffen.
    """
    vote_prompt = f"""
    Basierend auf dieser Persona:

    {persona_text}

    Welche Partei würde diese Person Ihrer Meinung nach eher wählen?
    Bitte nennen Sie nur den (fiktiven) Parteinamen oder eine kurze Begründung.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # oder GPT-4
            messages=[
                {"role": "user", "content": vote_prompt}
            ],
            max_tokens=50,
            temperature=0.5
        )
        predicted_vote = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Fehler bei Wahlvorhersage: {e}")
        predicted_vote = "Vote konnte nicht vorhergesagt werden."
    
    return predicted_vote

# -----------------------------------------------------------------------------
# 5) PIPELINE-AUSFÜHRUNG PRO ZEILE
# -----------------------------------------------------------------------------
# Neue Spalten anlegen
df["generated_persona"] = ""
df["predicted_vote_llm"] = ""

for idx in range(len(df)):
    row = df.iloc[idx]

    # 1) Persona generieren
    persona_text = generate_persona(row)
    df.at[idx, "generated_persona"] = persona_text

    # 2) Mit Persona das Wahlverhalten vorhersagen
    predicted_vote = predict_vote(persona_text)
    df.at[idx, "predicted_vote_llm"] = predicted_vote

# -----------------------------------------------------------------------------
# 6) SPEICHERN DER ERGEBNISSE
# -----------------------------------------------------------------------------
df.to_csv("personen_datensatz_mit_llm.csv", index=False)
print("Fertig! Die Spalten 'generated_persona' und 'predicted_vote_llm' wurden hinzugefügt.")
