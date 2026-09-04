Genera un’app per smartphone (orientamento verticale) con UI semplice, essenziale e altamente leggibile anche in condizioni di scarsa illuminazione. Usa caratteri e pulsanti della massima dimensione possibile compatibilmente con la struttura descritta.

📌 Obiettivo dell’app
L’app è un tool per musicisti dedicato allo studio dell’intonazione tramite riproduzione di sequenze monodiche (una nota per volta) di note.
L’utente inserisce una sequenza, imposta il tempo, e avvia la riproduzione.
L’app non deve analizzare l’intonazione tramite microfono in questa versione.

📌 Struttura generale della UI
La schermata è unica, divisa verticalmente in tre sezioni principali:

1. Display della sequenza (griglia 8×2)
Mostra 16 note disposte in una griglia 8 colonne × 2 righe.

Ogni cella rappresenta una nota monofonica della sequenza.

Ogni nota ha durata fissa: 1 beat (1/4).

Durante la riproduzione, la cella corrispondente alla nota attuale viene evidenziata.

Stile:

caratteri molto grandi compatibilmente con le altre specifiche date

contrasto elevato (es. sfondo scuro, testo chiaro)

evidenziazione con colore forte e ben visibile

2. Controlli di riproduzione
Pulsanti grandi, facilmente premibili, con etichette chiare:

Start — avvia la riproduzione della sequenza

Stop — ferma la riproduzione mantenendo la posizione corrente

Reset — riporta il puntatore alla prima nota

Loop ON/OFF — toggle per ripetere la sequenza in ciclo

Tempo + — aumenta il BPM

Tempo − — diminuisce il BPM

Range BPM: 40–200

Nessuna visualizzazione grafica del tempo: il BPM è solo un parametro interno

3. Inserimento della sequenza
Pulsanti grandi, disposti come una mini tastiera stile pianoforte, ottava centrale.

Ogni pressione aggiunge una nota alla sequenza (fino a 16).

Pulsanti di gestione:

Del — cancella l’ultima nota inserita

Clear — cancella l’intera sequenza

La sequenza è monofonica e omoritmica: tutte le note durano 1 beat.

📌 Comportamento della riproduzione
La sequenza viene riprodotta con un suono sintetico semplice:

forma d’onda: triangolare

nessun filtro

envelope:

Attack: 1/20 s

Decay: 0

Sustain: 100%

Release: 1/20 s (anticipato rispetto alla fine della nota)

La riproduzione segue la sequenza nota per nota, evidenziando la cella attiva.

📌 Workflow dell’utente
Inserisce le note tramite la tastiera (max 16).

Imposta il BPM (default 120).

Avvia la riproduzione con Start.

Se attivo, Loop ripete la sequenza all’infinito.

Può fermare con Stop o riportare all’inizio con Reset.

📌 Stile e design
UI minimalista, funzionale, senza elementi superflui.

Massima leggibilità:

pulsanti grandi

font molto grandi

contrasto elevato

layout ottimizzato per smartphone verticali

Nessuna animazione complessa: tutto deve essere immediato e chiaro.