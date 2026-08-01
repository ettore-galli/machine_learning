# SETUP OLLAMA LOCAL

## Download and place ollama locally

```txt
https://ollama.com/download/mac
```

## Create an environment

@see env.sh

## One time setup script

@see setup.sh

## Daily start script

@see ollama-


[copilot]

Sì, **puoi tenere i file GGUF su un drive esterno** e usarli con Ollama senza problemi.  
La cosa importante è **come** glieli fai vedere: Ollama accetta *percorsi assoluti* nel `Modelfile`, quindi puoi puntare direttamente al disco esterno.

---

## 🔧 Come usare un GGUF su un drive esterno con Ollama

### 1. Metti il tuo GGUF sul drive esterno

Esempio:  
Il tuo disco esterno è montato come:

```
/Volumes/BigDisk/
```

Metti il modello qui:

```
/Volumes/BigDisk/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
```

---

## 📄 2. Crea un Modelfile che punta al percorso assoluto

Esempio:

```
FROM /Volumes/BigDisk/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf

PARAMETER temperature 0.2
PARAMETER top_p 0.9
```

⚠️ Nota:  

- Il percorso deve essere **assoluto**, non relativo.  
- Ollama non ha problemi con dischi esterni, purché siano montati prima di avviare il modello.

---

## 🏗️ 3. Crea il modello in Ollama

```
ollama create qwen-ext -f /Volumes/BigDisk/models/Modelfile
```

Ollama **non copia** il GGUF nel suo store interno se usi `FROM` con percorso assoluto:  
lo usa direttamente dal disco esterno.

---

## 🧪 4. Esegui il modello

```
ollama run qwen-ext
```

---

## 📌 Considerazioni importanti

### ✔️ Funziona bene se

- Il drive esterno è **SSD**  
- È collegato via **USB‑C / Thunderbolt**  
- Il filesystem è **APFS** o **ext4** (evita FAT32/exFAT per file >4GB)

### ⚠️ Attenzione a

- **Tempi di accesso**: un HDD USB 2.0 può rallentare il caricamento iniziale.  
- **Montaggio del disco**: se il disco non è montato, Ollama non trova il GGUF.  
- **Permessi**: su macOS a volte serve dare permessi alla cartella del disco esterno.

---

## 🎯 Vuoi che ti preparo un Modelfile già pronto per il tuo modello GGUF e il tuo percorso sul disco esterno?

👉 Modelfile personalizzato
