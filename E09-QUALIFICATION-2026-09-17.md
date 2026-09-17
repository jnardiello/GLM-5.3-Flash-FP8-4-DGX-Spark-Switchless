# E09 — verifica del ragionamento e qualificazione

Esito al 17 settembre 2026, 20:11 UTC: **unresolved**. Low fallisce la correttezza
Go; High fallisce sia formato sia correttezza; Max raggiunge il budget nei due
campioni osservati prima dell'interruzione. Nessun profilo supera il proprio
pilot. La riattivazione del thinking non ha risolto l'anomalia e la causa
thinking-off resta non dimostrata. Il percorso si ferma prima della conferma
e della transizione DFlash. I quattro rank sani rimangono caricati.

## Ambito e protocollo effettivo

Il proprietario ha precisato che questa fase verifica la validità delle risposte
e che la conversazione concorrente era la propria. Tale traffico è ammesso
durante i pilot: tempi e throughput nelle ricevute non qualificano prestazioni.
Rimangono attivi i controlli di salute, identità, sorgenti e trasporto dei quattro
rank. Una successiva serie prestazionale richiederebbe esclusività.

Ogni invocazione completa usa il comando nativo Rigmark, cinque campioni codice,
cinque prosa e cinque structured, senza prefill o concorrenza. Prompt, nonce,
comparison ID `glm53-nq2-3x-20260911-v1`, seed `20260905`, temperatura 0,
top_p 1 e budget totale 8192 sono invariati. Label e cache salt sono nuovi.
Il modello target-only conserva i processi caricati prima della diagnosi.

Il piano approvato adotta pilot prima della conferma: preferire Low, poi High,
poi Max soltanto se il rispettivo pilot passa. Servono cinque risposte codice
valide e i controlli prosa/structured; poi tre invocazioni nuove, codice 15/15,
prima di correggere e caricare la candidata DFlash. I pilot non entrano in
eventuali mediane della qualificazione.

## Risultati disponibili

| Braccio | Richieste | Formato codice | Compilazione | Test del modello | Indipendenti | Esito |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| true / Max, tentativo interrotto | 2 completate, 1 interrotta | Non auditabile | Non eseguita | Non eseguiti | Non eseguiti | Due risposte al cap 8192; nessun pilot valido |
| true / Low | 15 | 5/5 | 5/5 | 2/5 | 0/3; due firme non supportate | Non qualificato |
| false / Low | 15 | 5/5 | 4/5 | 4/4 eseguiti | 0/2; due firme non supportate, un caso non compilato | Controllo negativo |
| true / High | 15 | 3/5 | 4/5 | 3/4 eseguiti | 1/4; un caso non compilato | Non qualificato |

Entrambi i run Low passano i 15 gate nativi di base, ma quei gate non verificano
la correttezza Go. Tutti i dieci output codice Low finiscono con `stop`, sono
visibili nel contenuto e registrano zero caratteri reasoning. Questo è un dato
delle ricevute, non una prova che i due percorsi interni siano equivalenti.

True/Low: il caso 2 produce `panic: sync: negative WaitGroup counter`; il caso 3
pretende zero token quando ne restano due; il caso 5 si aspetta il rifiuto di
un consumo ancora possibile. False/Low: il caso 5 non compila perché i test
usano `base` fuori dal suo scope. Sono errori del deliverable, indipendenti
dalle limitazioni degli adattatori dell'oracolo.

True/High: il caso 3 aggiunge la prosa `*(Word count: ~950)*`, benché superi tutti
i test Go e il race detector; il caso 5 contiene 1120 parole e non compila, perché
`1e308 * 10` è una costante che eccede float64. Il caso 4 pretende nei propri
test che un clock nil sia accettato, mentre il costruttore lo rifiuta. Gli
indipendenti trovano inoltre che il suo clock arretrato, tornando all'istante
precedente, accredita due volte il refill. I casi 1/2 passano i propri test ma
accettano NaN/Inf. Nessuno dei cinque passa tutti i requisiti insieme.

In High il reasoning è separato e non vuoto per 5/5 risposte (504, 99, 598, 89,
656 caratteri), mentre il codice finale è visibile e termina con `stop` in
tutti i casi. Il budget non viene esaurito. Il thinking effettivamente attivo
non basta quindi a correggere la consegna entro i vincoli concordati. Non è un
confronto false/true a pari High: quel controllo non è stato avviato dopo il
fallimento del candidato e l'applicazione del criterio di arresto.

Tutti gli indipendenti eseguibili Low falliscono sul costruttore che accetta
NaN e infinito positivo; in High passano soltanto per il caso 3. Le esecuzioni
con `-race` falliscono gli stessi assert semantici: non ne deriva
che sia stata osservata una data race. Le firme non supportate sono riportate
come inconcludenti, senza modificare il codice generato o fingere un PASS.
Il riepilogo JSON conservativo può dire `inconclusive` quando coesistono firme
non supportate e fallimenti certi: leggere gli esiti distinti delle fasi.

Il primo true/Max fu fermato erroneamente applicando l'esclusività anche alla
diagnosi; il proprietario ha poi chiarito il traffico. Due richieste avevano
già raggiunto 8192 token, la prima prima della conversazione concorrente.
Il runner nativo salva la ricevuta al termine: l'interruzione ha lasciato solo
il log di avanzamento, non i testi o un audit riproducibile. Non attribuiamo
quei token al reasoning e non trattiamo il tentativo come invocazione completa.
Il controllo false/Max non è stato eseguito. Questo limita il confronto causale
a Max e viene mantenuto come deviazione, senza sostituire dati mancanti.

Conteggio effettivo: **tre invocazioni complete, 45 richieste**, più tre richieste
Max avviate, di cui due completate per lunghezza e una interrotta lato client.
Sono quindi 48 richieste diagnostiche avviate e 47 completamenti osservati dal
client, con 15 risposte codice conservate nelle ricevute complete. I contatori
di servizio includono anche le conversazioni del proprietario. Zero nuove
richieste di gate/API, zero conferme, zero run di qualificazione DFlash. Il piano
da 330/360 richieste non è stato completato perché i criteri qualitativi falliscono.
False/High e false/Max restano non misurati; nessuna coppia o mediana mancante
viene ricostruita. I cinque nonce ripetono lo stesso task, non cinque problemi
indipendenti.

## Strumenti corretti e verifiche

Il checkout completo Rigmark è stato recuperato in
`~/workspace/jacopo/rigmark`, preservando Git e modifiche preesistenti. La copia
originaria, il manifest e gli hash sono conservati nella cartella privata
`~/E09-QUALIFICATION-20260917`. Le nuove prove Low e successive usano un'unica
versione fissata; il tentativo Max precedente conserva separatamente il proprio
client preliminare.

- `audit-code` riconosce fence Markdown e nomi file nel fence, nel commento
  iniziale o in un'etichetta adiacente; esige due file distinti, blocchi chiusi,
  nessuna prosa e meno di 1100 parole separate da spazi. Due file identificabili
  vengono eseguiti anche se il formato fallisce; duplicati o ambiguità non sono
  risolti scegliendo arbitrariamente.
- Il JSON `--output` collega la ricevuta immutata al suo SHA, separa consegna,
  formato, compilazione, test del modello, oracolo e race detector. Registra
  adattatore esplicito, hash dei sorgenti e immagine Docker immutabile.
- `--section decode|concurrency` seleziona gli output da auditare; il default
  rimane decode. `run --skip-decode`, disattivo per default, consente il futuro
  run nativo solo C1/C2/C4 con `--skip-prefill`. Ricevute e report distinguono
  sezioni omesse dai controlli passati. I vecchi gate mantengono il significato.
- I test del codice generato usano Docker senza rete, filesystem e sorgenti in
  sola lettura, limiti di CPU/memoria/processi, nessuna capability, toolchain e
  dipendenze senza download. Errori ambiente, timeout e fallimenti sono separati.

Sono passati **98 test Rigmark**, inclusi i controlli Docker reali, il controllo
positivo con race detector, le regressioni sui fence, il replay dei difetti Go
della RCA e il run nativo simulato con 21 richieste concorrenti senza decode.
Il primo replay Docker delle fixture aveva un errore di permessi, corretto
copiando le fixture sotto l'utente del runner e fissando UID:GID nel container;
non è stato conteggiato come errore del modello.

Un controllo runtime eseguito mentre era presente un container CPU dell'audit
ha segnalato una seconda istanza Docker. Ripetuto dopo la chiusura dell'audit,
passa con gli stessi quattro processi serving: quel FAIL riguarda la sovrapposizione
dei controlli, non un secondo motore o un restart.

Passano anche `scripts/check.sh`, il controllo dei link Markdown e
`git diff --check` nei due checkout. Le modifiche del client rispetto alla copia
completa originaria sono registrate separatamente, preservando il lavoro già
presente al momento del recupero.

Il nuovo audit della ricevuta storica target-only, lasciata immutata, dà formato
2/5, compilazione 4/4 eseguibili, test del modello 3/4, indipendenti 0/4. Il caso
5 duplicato non è eseguito; entrambe le versioni conservate sono invece coperte
dalle regressioni forensi, senza scegliere una versione per qualificare il run.

## Pin, evidenze e stato del servizio

| Artefatto | SHA-256 |
| --- | --- |
| Archivio client originale | `0b5b57f93d4f78703ee00d42cf6c9e8985cb3375eaae31786e805f4c8a75450e` |
| Client fissato per Low e successivi, sorgenti | `82abcd0817e7d74e7fa2d825fe1a2c05e7315029a2e0f3e05ce3edd154fd666c` |
| Client fissato, worktree | `08276f528923ba5b5043f9090abb0f40662ccb0e08c412b5a65d97d04e1d5bb7` |
| Prompt invariati | `0c3ac4015813af6a1b967659bd2289990fde895a5e43ae065b894d1e1f909ecb` |
| Immagine Go, digest registry | `699337d620559a59b4a2bb298ad59611e535d2ee755a34cf2d2a98f37578dc80` |
| Immagine Go, ID locale | `4e2219ca515d99610115539a96f733fa567f62941e31ac5bfb87c7ade6143bcc` |
| Ricevuta true/Low | `59a5744b2ccdc6a2aab65e3e4d40d47bf8bb04d41c2f85e0773615d22e2e3360` |
| Ricevuta false/Low | `ecf6b74d7d329bd60e52fab917cd396e30c3682fcdc178383b1a317e065caadd` |
| Ricevuta true/High | `97930153488e402bc040d3895c9aa88f33615bfe7772abd32209c29ef5beeb8f` |
| Audit true/Low | `7b26d95b24d01c8d1c7ec9a874fe1ef0196bd01efc63aa2efb9fca4b5da121d3` |
| Audit false/Low | `72dda8d58b8a271a583714abd1e52a599074321082401ed4741f89db7f73a1d6` |
| Audit true/High | `7c2dc322efbd98f1c2b9a645be37d1b270995680f1d86bea735a69ca4ae865a9` |
| Pacchetto delle evidenze remote | `a91e74427c2ac20a1279de7e344642558072b72e52472c380557056e74c29f64` |

Le prove remote sono nella nuova finestra
`~/tp4/candidate-windows/e09-reasoning-qualification-20260917`; quelle locali in
`~/E09-QUALIFICATION-20260917`. Le finestre sigillate precedenti restano immutate.
I file privati sono tenuti fuori dal checkout.
Ricevute e audit copiati sono identici byte per byte ai file remoti; il manifest
locale `SHA256SUMS` include log di verifica, snapshot e sorgenti conservati.

Il controllo finale alle **20:11:27 UTC** passa sorgenti e identità dei quattro
rank, health 200, zero restart, idle 0/0; contatore 87 che include le conversazioni
del proprietario. I quattro client diagnostici sono conclusi, senza container
di audit residui. Nessun reload in questa diagnosi. L'overlay
attivo rimane `scripts/node/etc/local/e09-r10-on-target-only-20260917.env`, SHA
`037361b34a39f83fbb7fb912dffe4e7e8aa1fead1e8cca7a024192161683cc19`.
Per un eventuale stop coordinato usare lo stesso overlay e il controller fissato
`tp4ctl-e04-window-6cb6f07c`, SHA
`6cb6f07cc60a13c86dfed8c01156c13d3f1fb88ca611bd499989841f96688a82`.
Nessun ripristino automatico F0/E07a, commit, push o promozione.

## Fasi non avviate e confronto storico

Non è stato scelto un effort predefinito. La modifica del template/API con HTTP
400 per false e `none`, i gate Roma/tool aggiornati, le verifiche multi-turn e
la nuova ricetta DFlash erano subordinati alla conferma del profilo: non sono
stati applicati o qualificati. La ricetta target-only e gli artefatti legacy
rimangono intatti; non occorre rollback delle modifiche al servizio, perché
non ce ne sono state. La sola transizione DFlash autorizzata non è stata usata.

I tre run standard da 54 richieste e i tre run di codice concorrente completo da
21 richieste ciascuno non sono iniziati. Le risposte storiche C1/C2/C4 a 256 token
non rappresentano programmi completi. Non esistono nuove mediane confrontabili
con [F0 congelata](docs/baseline-f0.json); il thinking modifica inoltre il lavoro
svolto. Nessuna conclusione prestazionale si ricava dai pilot con traffico ammesso.
F0 conserva tre run e 162 richieste; la variante ha **0/3 run standard e 0/162
richieste**, più **0/3 run di codice concorrente completo e 0/63 richieste**.
La variabilità F0 per run rimane nel JSON congelato; non esiste una nuova
variabilità prestazionale qualificata da riportare.

| Metrica | F0, mediana congelata | Variante, mediana di tre run |
| --- | ---: | --- |
| Decode codice, token/s | 50.401 | Non misurata |
| C1 storico 256 token, token/s | 37.223 | Non misurata |
| C2 storico 256 token, token/s | 57.009 | Non misurata |
| C4 storico 256 token, token/s | 71.084 | Non misurata |
| Decode prosa, token/s | 29.220 | Non misurata |
| Codice primo token, s | 0.410 | Non misurata |
| Prosa primo token, s | 0.397 | Non misurata |
| C1 primo token, s | 0.404 | Non misurata |
| C2 primo token, s | 0.616 | Non misurata |
| C4 primo token, s | 0.935 | Non misurata |
| Prefill 8K cold, token/s | 2112.218 | Non misurata |
| Prefill 8K replay, token/s | 4649.988 | Non misurata |
| Prefill 32K cold, token/s | 2201.970 | Non misurata |
| Prefill 32K replay, token/s | 21524.192 | Non misurata |
| Prefill 64K cold, token/s | 2201.282 | Non misurata |
| Prefill 64K replay, token/s | 35971.377 | Non misurata |
| Prima risposta visibile | Non riportata negli aggregati congelati | Non qualificata |
| Consegna programmi completi concorrenti | Nessun riferimento storico | Non misurata |

## Diagnosi successiva proposta, non avviata

La correzione dei controlli è verificata; la causa della generazione rimane
aperta. Un solo programma High che supera gli indipendenti dimostra che questo
runtime sa produrre una soluzione funzionante per il task, non che il motore sia
numericamente corretto in tutte le posizioni. Allo stesso modo, test generati
errati o prosa vietata non dimostrano da soli un guasto numerico.

Il passo mirato proposto è preservare i prefissi dei casi difettosi e quelli del
caso High che supera gli indipendenti. Nel controllo numerico successivo,
confrontare le stesse sequenze teacher-forced via prefill e decode incrementale,
registrando solo finestre limitate intorno alla prima divergenza: token scelto,
top logits e margine, posizioni KDA/sparse attention e slot KV. Fissare tolleranze
FP8 esplicite e cercare divergenze ripetibili; se non emergono, mantenere aperte
fallibilità del modello e sampling greedy. Non cambiare contemporaneamente
sampling, kernel, connettore o modello. Nessuna traccia live aggiuntiva o variante
del motore è stata lanciata in questa fase.
