# Roadmap 4 settimane — Diventare operativo con OpenAI Assistants

Percorso pratico, orientato alla costruzione di un portfolio AI professionale.

---

## 🗓️ Settimana 1 — Fondamenta operative degli Assistants

**Obiettivo:** capire struttura, threads, runs, messaggi, file, modelli.

### Cosa studiare

- Documentazione ufficiale OpenAI Assistants (gratuita)  
  <https://platform.openai.com/docs/assistants/overview>  
- Concetti base LLM (gratuito, HuggingFace)  
  <https://huggingface.co/learn/nlp-course/chapter1/1>  
- Prompt engineering (gratuito, OpenAI)  
  <https://platform.openai.com/docs/guides/prompt-engineering>  
- Introduzione ai modelli GPT‑4.1 / GPT‑4o  
  <https://platform.openai.com/docs/models>

### Cosa fare (pratica)

- Creare un Assistant con istruzioni personalizzate  
- Creare un thread, inviare messaggi, leggere risposte  
- Caricare un PDF e chiedere un riassunto  
- Salvare e riprendere un thread da file JSON  
- Testare differenze tra modelli (4.1, 4o, mini)

### Mini‑progetto della settimana

#### **Assistant “Analizzatore di PDF”**  

Carichi un PDF → l’assistente lo legge → produce un riassunto strutturato.  
Include:

- parsing PDF  
- thread persistente  
- output JSON strutturato

### Risorse utili

- Parsing PDF in Python (GeeksForGeeks)  
  <https://www.geeksforgeeks.org/how-to-extract-text-from-pdf-file-using-python/>
- Esempi Assistants API (GitHub)  
  <https://github.com/openai/openai-python/tree/main/examples>

---

## 🗓️ Settimana 2 — Tool Calling + Funzioni Python

**Obiettivo:** far usare strumenti reali all’assistente.

### Cosa studiare

- Tool calling (OpenAI docs)  
  <https://platform.openai.com/docs/assistants/tools>  
- JSON Schema per funzioni  
  <https://json-schema.org/learn/getting-started-step-by-step.html>  
- API REST in Python (GeeksForGeeks)  
  <https://www.geeksforgeeks.org/rest-api-introduction/>

### Cosa fare (pratica)

- Definire funzioni Python come strumenti  
- Validare parametri con JSON Schema  
- Implementare tool che chiama API esterne (meteo, GitHub, ecc.)  
- Logging delle chiamate e degli argomenti generati dal modello  
- Gestire chiamate multiple in un singolo run

### Mini‑progetto della settimana

#### **Assistant “CRM Query Bot”**  

L’assistente risponde a domande sui clienti usando funzioni Python che interrogano un database SQLite.

Include:

- tool Python → query SQL  
- risposta ragionata dell’assistente  
- gestione errori (cliente non trovato, ID errato)

### Risorse utili

- SQLite in Python (GeeksForGeeks)  
  <https://www.geeksforgeeks.org/sql-using-python/>
- API GitHub (gratuito)  
  <https://docs.github.com/en/rest>

---

## 🗓️ Settimana 3 — RAG nativo + File + Code Interpreter

**Obiettivo:** costruire applicazioni AI che usano documenti, dataset e codice.

### Cosa studiare

- Retrieval nativo negli Assistants  
  <https://platform.openai.com/docs/assistants/tools#retrieval>  
- Code Interpreter  
  <https://platform.openai.com/docs/assistants/tools#code-interpreter>  
- Introduzione al RAG (HuggingFace, gratuito)  
  <https://huggingface.co/learn/cookbook/retrieval-augmented-generation>

### Cosa fare (pratica)

- Caricare una cartella di documenti e fare Q&A  
- Estrarre tabelle da PDF  
- Usare Code Interpreter per analizzare un CSV  
- Generare grafici e salvarli come file  
- Creare un report finale (Markdown o JSON)

### Mini‑progetto della settimana

#### **Assistant “Analista Dati Automatico”**  

Carichi un CSV → l’assistente esegue analisi → produce grafici → genera un report.

Include:

- Code Interpreter  
- grafici Matplotlib/Seaborn  
- generazione file  
- spiegazione dei risultati

### Risorse utili

- Analisi dati con Pandas (GeeksForGeeks)  
  <https://www.geeksforgeeks.org/python-pandas-tutorial/>
- Matplotlib (GeeksForGeeks)  
  <https://www.geeksforgeeks.org/matplotlib-tutorial/>

---

## 🗓️ Settimana 4 — Orchestrazione avanzata + Deployment

**Obiettivo:** costruire un’applicazione AI completa, robusta, con workflow complessi.

### Cosa studiare

- Pattern agentici (HuggingFace)  
  <https://huggingface.co/learn/cookbook/agents>  
- FastAPI (gratuito)  
  <https://fastapi.tiangolo.com/>  
- Logging e audit trail (Python logging)  
  <https://docs.python.org/3/howto/logging.html>  
- Deployment su server (GeeksForGeeks)  
  <https://www.geeksforgeeks.org/deploying-fastapi-app-on-ubuntu/>

### Cosa fare (pratica)

- Assistant che decide quale tool usare in base al contesto  
- Assistant che combina RAG + tool + code interpreter  
- Esposizione via API REST (FastAPI)  
- Persistenza dei thread in database  
- Gestione errori e fallback

### Mini‑progetto finale

#### **AI Workflow Assistant completo**

Un sistema che:

1. Riceve un documento  
2. Lo analizza (RAG)  
3. Estrae dati (tool)  
4. Li elabora (code interpreter)  
5. Aggiorna un database o invia un’email (tool)  
6. Produce un report finale

### Risorse utili

- FastAPI + OpenAI (GitHub)  
  <https://github.com/openai/openai-python/tree/main/examples>  
- Esempi di orchestrazione agentica (HuggingFace)  
  <https://huggingface.co/learn/cookbook>

---

## 🎓 Risultato finale

Dopo 4 settimane avrai un portfolio con:

- Assistant per analisi PDF  
- Assistant con tool-calling e database  
- Assistant per analisi dati con Code Interpreter  
- AI Workflow Assistant completo (RAG + tool + API)

Spendibile nel mercato italiano per ruoli:

- AI Workflow Developer  
- AI Integration Specialist  
- AI Automation Engineer  
- Data & AI Analyst

---

## 🔗 Guided Links utili

- [Struttura Assistant](ca://s?q=Spiegami_struttura_OpenAI_Assistant)
- [Tool calling](ca://s?q=Voglio_capire_tool_calling)
- [RAG con Assistants](ca://s?q=Studiare_RAG_con_Assistants)
- [Code Interpreter](ca://s?q=Capire_Code_Interpreter)
- [Threads e Runs](ca://s?q=Come_funziona_threads_runs)
- [Document AI Assistant](ca://s?q=Costruire_Document_AI_Assistant)
- [AI Workflow Automation](ca://s?q=Costruire_AI_workflow_automation)
- [Analisi dati AI](ca://s?q=Costruire_AI_analisi_dati)
