# LM Studio (lms) da riga di comando su Mac (versione 0.4.18+)

[Prodotta da Grok]

## Installazione / Primo Avvio

`lms` è già incluso in LM Studio.  

1. Apri LM Studio almeno una volta (importante).  
2. Apri **Terminale** e digita:  

   ```bash
   lms --help
   ```

   (Dovrebbe mostrare i comandi disponibili. Se non funziona, riavvia il Terminale o prova `~/.lmstudio/bin/lms`).

## Comandi Principali

- **Stato generale**:  

  ```bash
  lms status
  ```

- **Server**:
  - Avvia il server:  

    ```bash
    lms server start
    ```

  - Avvia con opzioni comuni:

    ```bash
    lms server start --port 1234          # Porta specifica (default spesso 1234)
    lms server start --cors               # Abilita CORS (utile per web/app)
    lms server start --bind 0.0.0.0       # Rende accessibile sulla rete locale (attenzione sicurezza!)
    lms server start --port 1234 --cors --bind 0.0.0.0
    ```

  - Stato del server:  

    ```bash
    lms server status
    ```

  - Arresta il server:  

    ```bash
    lms server stop
    ```

- **Modelli**:
  - Lista modelli scaricati: `lms ls`
  - Modelli caricati in memoria: `lms ps`
  - Carica un modello: `lms load nome-modello`
  - Scarica un modello: `lms get autore/modello`

Altri comandi utili: `lms log stream` (per vedere i log), `lms --help` o `lms server start --help` per dettagli.

## Configurazione

- Il server usa per default le impostazioni dell'ultima sessione (porta, CORS, ecc.).
- Puoi sovrascrivere tutto con i flag (`--port`, `--cors`, `--bind`).
- Le configurazioni persistono in file interni (solitamente in `~/.cache/lm-studio/` o simili).
- Per esposizione sulla rete: usa `--bind 0.0.0.0` + **autenticazione** consigliata per sicurezza.

## Avvio e Arresto Manuale

**Avvio manuale** (consigliato per uso semplice):

1. Apri Terminale.
2. Esegui `lms server start` (con i flag desiderati).
3. Il server resta attivo finché non chiudi il Terminale o usi `stop`.

**Per tenerlo in background** (senza bloccare il Terminale):

```bash
nohup lms server start --port 1234 --cors > ~/lms-server.log 2>&1 &
```

O usa `screen` / `tmux`.

**Arresto manuale**:

```bash
lms server stop
```

**Script semplice** (es. `start-lms.sh`):

```bash
#!/bin/bash
lms server start --port 1234 --cors
```

Rendilo eseguibile con `chmod +x start-lms.sh`.

Per uso headless/server senza GUI c'è anche `llmster` (daemon), ma per la tua versione standard `lms server` è sufficiente.

**Note**:

- Il server espone un'API compatibile con OpenAI (`http://localhost:1234/v1/...`).
- Carica prima un modello (`lms load ...` o dalla GUI) se serve.
- Per problemi: controlla i log con `lms server status --verbose` o `lms log stream`.

Se hai bisogno di uno script specifico o configurazione avanzata (es. systemd-like su Mac con launchd), fammi sapere!

### Test

```shell
curl http://localhost:9876/v1/models
```
