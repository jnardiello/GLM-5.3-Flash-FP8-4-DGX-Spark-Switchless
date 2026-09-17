# Piano di ottimizzazione

Questo documento conserva il catalogo di esperimenti accettato dal proprietario. Non
autorizza interventi sul cluster, avvii, benchmark o promozioni: ogni esperimento richiede
una finestra operativa concordata e segue il flusso in [`AGENTS.md`](AGENTS.md). Gli esiti
possibili restano `promote`, `discard` o `unresolved`; le fonti sostengono le ipotesi, non
garantiscono miglioramenti.

Il modello pubblico descrive 45 layer, di cui 34 KDA e 11 MLA, MoE con 8 esperti attivi
su 288 e `mHC=4`; questi valori provengono dal
[`config.json` di GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json).
La priorità di valutazione rimane quella del repository: codice C1 e concorrenza C2/C4,
poi prosa e prefill, sempre con correttezza, integrità degli errori e quattro rank sani.
Non sono previste percentuali di guadagno a priori.

Il protocollo ordinario usa F0 come baseline congelata senza rieseguirla. Ogni variante
esegue tre run Rigmark nativi consecutivi, con un solo caricamento dei pesi e lo stesso
processo per tutti e tre; il confronto usa la mediana dei tre valori per-run contro la
mediana F0 congelata. Solo una successiva indicazione esplicita del proprietario può
cambiare questo conteggio per un esperimento futuro.

## Catalogo

### E01 — Modalità `batch-uniform` dello scheduler adattivo

- **Modifica:** passare `VLLM_ADAPTIVE_K_MODE` da `per-request` a `batch-uniform`,
  lasciando invariati `k_lo=3`, `k_hi=5` e ogni altro parametro.
- **Metriche attese:** verificare soprattutto throughput e TTFT di codice C1/C2/C4 e
  decode codice/prosa; l'ipotesi è che gli step uniformi usino più spesso i CUDA graph
  FULL invece del percorso PIECEWISE, senza assumere un guadagno.
- **Fonte:** descrizione dei due percorsi in
  [`adaptive_k_scheduler.py`](scripts/node/patches/adaptive_k_scheduler.py#L47-L66).
- **Stato:** scartato per decisione del proprietario il 2026-09-12. Codice e prosa sono
  stati giudicati entro il rumore; lo scarto non attribuisce una regressione dimostrata a
  C4. Un run nativo è terminato con 54 richieste valide, zero errori di validazione e gate
  codice 5/5; il secondo è stato interrotto dal proprietario e il terzo non è iniziato.
  Non esiste quindi una mediana di tre run. L'evidenza resta nell'archivio esterno.

### E02 — Precompilazione e warmup JIT mHC

- **Modifica:** precompilare e riscaldare i kernel JIT mHC necessari prima della misura,
  senza cambiare il calcolo servito durante il run.
- **Metriche attese:** stabilità del primo uso misurato, TTFT e throughput prefill
  cold/replay; controllare che decode, concorrenza ed errori non regrediscano.
- **Fonti:** guida vLLM al
  [JIT kernel warmup](https://docs.vllm.ai/en/latest/contributing/jit_kernel_warmup/)
  ed evidenza operatore NQ4 non riproducibile pubblicamente, righe log 436–439,
  conservata nell'archivio esperimenti senza esporne il percorso privato.
- **Stato iniziale:** da testare.

### E03 — Prefill mHC token-sharded R33

- **Modifica:** applicare il percorso R33 token-sharded al prefill mHC, distribuendo le
  8192 righe globali come 2048 righe per rank; procedere solo dopo un port FP8 verificato.
- **Metriche attese:** throughput prefill cold/replay a 8K, 32K e 64K, TTFT, memoria ed
  errori; codice, concorrenza e prosa restano gate di non regressione.
- **Fonte:**
  [GLM53 TP4 prefill quickstart di SparkRing](https://github.com/FujitsuPolycom/sparkring/blob/main/docs/GLM53_TP4_PREFILL_QUICKSTART.md).
- **Stato iniziale:** da testare.

### E04 — CUDA graph DFlash2 con forme esatte

- **Modifica verificata:** rispetto a F0, aggiunti soltanto
  `--cudagraph-capture-sizes 1 2 4 8 10 12 14 16 18 20 22 24 32 40 48 56 64 72`
  e `--max-cudagraph-capture-size 72`. La prova statica del dispatcher e delle mappature,
  il controllo F0 successivo all'unico `systemctl daemon-reload` sul rank 3 e l'identità
  runtime E04 sui quattro rank hanno dato PASS.
- **Esecuzione:** E04 ha caricato i pesi una volta e raggiunto `/health` 200. L'invocazione
  remota con l'intero sorgente del controller in `bash -c` è stata intercettata dal proprio
  `flusher_off`; inoltre il relay locale aveva il blocco Python dei gate in posizione errata
  ed è terminato con `NameError`. Il termine dei gate è scaduto, quindi E04 è stato arrestato
  prima dei gate e di Rigmark: 0/3 run e 0/162 richieste pianificate.
- **Primo tentativo:** `unresolved — non benchmarked`. Non esistono mediana o delta E04; l'errore
  operativo non costituisce evidenza favorevole o contraria alla variante CUDA graph e non
  autorizza promozione o scarto per prestazioni. Dopo la verifica dello stato fermo su tutti
  i rank, il proprietario ha autorizzato un retry E04 diretto. Il retry ha superato preflight,
  identità runtime e gate funzionali, poi ha completato due run nativi e 108 richieste senza
  errori di validazione. R1 è l'unico run prestazionalmente pulito; durante R2 sono stati
  osservati separatamente sei POST chat estranei non loopback o non identificabili e quattro
  completion riuscite oltre il conteggio nativo atteso, senza poterli associare uno a uno.
  R2 resta valido per correttezza ma viene escluso dal confronto delle prestazioni. Altre
  quattro completion inattese sono arrivate durante la pausa. Il solo R1 pulito ha mostrato
  codice lungo 50.414 contro 50.401 di F0, C1 37.143 contro 37.223, C2 50.975 contro
  57.009, C4 69.772 contro 71.084 e prosa 28.451 contro 29.220. Non esiste una mediana E04:
  l'esito prestazionale resta `unresolved`. Il proprietario ha concluso che E04 non dimostra
  benefici da mantenere, ha chiuso l'esperimento senza promozione e saltato R3 e una nuova
  serie. Il lifecycle di ripristino F0, con un caricamento pesi separato, ha superato i gate
  funzionali entro 25 secondi dal primo `/health` 200; anche il controllo operativo F0 finale
  ha dato PASS. F0 è ripristinato e verificato.
- **Ricevute:** i JSON nativi immutabili R1/R2 incorporano lo snapshot metadata precedente
  ai run (`native_runs_actual=0`, `measurement_in_progress`); i conteggi finali risiedono nel
  metadata E04 e nelle ricevute di validazione separate, senza riscrivere i JSON nativi.
- **Fonte:** record SparkRing sui
  [graph DFlash2 esatti](https://github.com/FujitsuPolycom/sparkring/blob/main/performance/records/glm53-flash/dflash2-exact-concurrency-graphs-20260904.md).
  Il record riguarda soprattutto NVFP4 e concorrenza C6+, quindi non predice il risultato
  della lane TP4 FP8.

### E05 — MTP nativo con profondità 3

- **Modifica verificata:** DFlash2 e lo scheduler adattivo sono stati sostituiti dal
  proposer MTP nativo a profondità 3 con riduzione argmax locale, mantenendo checkpoint
  target FP8, immagine, TP4 e il resto della ricetta F0.
- **Metriche attese:** throughput e TTFT decode codice/prosa, C1/C2/C4, accettazione dei
  token speculativi, qualità e memoria.
- **Fonte:** API vLLM per
  [GLM5Next MTP NVIDIA](https://docs.vllm.ai/en/latest/api/vllm/models/glm5next/nvidia/mtp/).
- **Risultato:** tre run nativi completi, 162 richieste, zero errori nativi, gate codice
  15/15 e nessun cap sul codice lungo, con un solo caricamento e lo stesso processo. La
  mediana codice lungo è 46.135 token/s contro 50.401 di F0; C1 38.943 contro 37.223;
  C2 55.710 contro 57.009; C4 82.378 contro 71.084; prosa 29.398 contro 29.220. C4 ha
  mostrato variabilità tra run, mantenuti tutti nel confronto previsto. C1/C2/C4 usano
  al massimo 256 token per agente, quindi il risultato C4 non dimostra un vantaggio
  sulle generazioni parallele lunghe.
- **Stato:** `discard` per l'uso generale corrente, per decisione del proprietario il
  2026-09-12; F0 resta la ricetta scelta. La decisione privilegia F0 dopo le perdite
  misurate nelle mediane del decode codice lungo e di C2, pur conservando i guadagni C1
  e C4 come evidenza. Non stabilisce che MTP sia sempre svantaggioso e non sono previste
  altre prove E05. Il report completo e le ricevute restano nell'archivio privato E05;
  il servizio F0 e i gate sono ripristinati, ma il controllo operativo completo resta
  non verde per il solo `NeedDaemonReload=yes` dell'unità fabric sul rank 3. Non è stata
  eseguita alcuna riparazione e causa e insorgenza non sono stabilite.

### E06 — Budget prefill da 8192 a 16384 token

- **Modifica:** aumentare soltanto il budget massimo di token batched/prefill da 8192 a
  16384, lasciando invariato il resto della ricetta.
- **Metriche attese:** throughput prefill cold/replay 8K/32K/64K, TTFT, C1/C2/C4,
  utilizzo memoria e stabilità.
- **Fonte:** guida vLLM al
  [tuning del chunked prefill](https://docs.vllm.ai/en/latest/configuration/optimization/#performance-tuning-with-chunked-prefill).
- **Stato iniziale:** da testare.

### E07 — Trasporto SparkRing SIRCL/RoCE con virtual mesh nativo

- **Modifica:** valutare il trasporto SIRCL/RoCE e il virtual mesh nativo di SparkRing
  sulla topologia fisica switchless corrente, al posto del percorso collettivo attuale.
- **Metriche attese:** throughput C2/C4 e prefill, TTFT, latenza dei collettivi, errori di
  trasporto e salute coerente dei quattro rank.
- **Fonte:** repository
  [FujitsuPolycom/sparkring](https://github.com/FujitsuPolycom/sparkring).
- **Stato:** E07a è stato eseguito ma resta `unresolved` e non promuovibile.
  Il riferimento F0 resta congelato nell'archivio privato opaco `f0-20260912-v4`, il cui
  manifest ha SHA-256
  `670a1877278108112375cdc029ae005aaf49a84b0f43f7c2c616333bc3369b9a`.
  Al momento della cattura, verifica offline, confronto live sui quattro rank e
  materializzazione della sorgente avevano dato PASS, mentre il controllo operativo era
  non verde per `NeedDaemonReload=yes` sul rank 0; la cattura non aveva eseguito reload,
  restart, ripristino, inferenze o benchmark. Un successivo `daemon-reload` autorizzato e
  verificato ha risolto quella differenza e il controllo operativo F0 ha dato PASS. Il
  proprietario ha scelto di non provare il ripristino reale prima del test E07a.
- **E07a:** candidato iniziale direct-ring SIRCL single-rail per il solo sync prefill, non
  virtual mesh ASIC. Usa SparkRing
  `b358a818786d8506086aaaabb9afe464fa2ccb49`, vLLM
  `487ecf187d3dfe74d2cf6119a92881dba403c219` e
  `libspark_transport_capi.so` SHA-256
  `f53c88b4bf885533c4d6de60dae1a9e46cf26db0695b37265955b7cc74934119`.
  Il percorso nativo accetta soltanto tensori BF16 contigui `[Q,4096]` con
  `Q=1024/2048/4096/8192`; `Q=128`, CUDA graph, decode e firme diverse restano sul
  percorso NCCL F0. Modello, immagine, DFlash2, scheduler, rete host e ricetta NCCL F0
  restano invariati salvo il payload e l'overlay di avvio SIRCL.
- **Gate numerico:** PASS in 359,213 s su 20/20 casi esatti: 4 fallback NCCL a `Q=128`,
  16 casi SIRCL nativi alle quattro Q ammesse e rifiuto atteso del voto incompatibile su
  4/4 rank; cleanup PASS. L'avvio ha poi superato i gate risposta coerente e tool-call
  2/2 entro 8,988 s dal primo `/health` 200 e ha provato una sessione SIRCL nativa
  `[8192,4096]` su 4/4 rank.
- **Rigmark:** completati tre run nativi consecutivi, 162/162 richieste, sali distinti,
  un solo caricamento dei pesi e lo stesso processo sui quattro rank. Non ci sono stati
  errori di `receipt.validate_result`; prose e structured hanno superato 15/15 gate.
  Nel quinto sample code di R3 il modello ha raggiunto il limite di 8192 token con
  `finish_reason=length`: code 14/15, un cap su 15 sample. La correttezza obbligatoria
  non è quindi soddisfatta; non ci sono prove che attribuiscano il cap a SIRCL.
- **Prestazioni diagnostiche:** le mediane dei tre valori per-run, non qualificabili a
  causa del gate fallito, sono: code 50.513 contro F0 50.401 token/s; C1 38.911 contro
  37.223; C2 53.891 contro 57.009; C4 73.481 contro 71.084; prosa 28.769 contro 29.220.
  Le TTFT sono code 0.412/0.410 s, C1 0.410/0.404, C2 0.646/0.616, C4 0.920/0.935 e
  prosa 0.393/0.397 (E07a/F0). I prefill cold/replay sono 2128.427/4709.914 a 8K,
  2205.697/20007.803 a 32K e 2212.390/35866.017 a 64K, contro F0
  2112.218/4649.988, 2201.970/21524.192 e 2201.282/35971.377 token/s.
- **Esito:** il fallimento del gate e i risultati primari non univoci lasciano E07a
  `unresolved / non promuovibile`. Il proprietario ha scelto di mantenerlo come runtime
  sperimentale sano durante l'integrazione locale di SparkCache, senza promuoverlo e senza
  ripristinare F0. F0 resta il riferimento congelato per ogni confronto futuro.

### E08 — Coalescenza del prefill di continuazione

- **Modifica:** coalescere i prefill di continuazione solo se un workload dedicato e
  riproducibile dimostra richieste di continuazione sufficienti; l'eventuale supporto di
  misura appartiene a Rigmark, senza wrapper locali.
- **Metriche attese:** TTFT e throughput del workload di continuazione, numero di prefill
  coalesciuti, memoria ed errori; la suite ordinaria resta un gate di non regressione.
- **Fonte:**
  [GLM53 TP4 prefill quickstart di SparkRing](https://github.com/FujitsuPolycom/sparkring/blob/main/docs/GLM53_TP4_PREFILL_QUICKSTART.md).
- **Stato iniziale:** da testare.

### E09 — Persistenza ibrida dei prefissi con SparkCache

- **Modifica:** provare la persistenza ibrida dei prefissi di SparkCache in uno scenario
  dedicato con prefissi condivisi. La compatibilità della lane FP8 non è ancora
  qualificata e deve essere verificata prima della misura.
- **Metriche attese:** throughput e TTFT del replay di prefissi, hit/reuse corretti,
  memoria e integrità delle risposte; lo scenario dedicato non qualifica da solo la
  variante per la produzione generale.
- **Fonte:** ricetta
  [SparkCache](https://github.com/FujitsuPolycom/sparkring/blob/main/recipes/sparkcache/README.md).
- **Stato:** dopo lo studio locale della sorgente SparkCache fissata a
  `f220230a5a85b94af8a296187241b6aacc3ed724`, il proprietario ha autorizzato la prova del
  build pubblicato JJ R10
  `ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:0d4029b3b7023cf32c37ac20279469c9a2ee16a057f25aae3bcfee9ee5fb660f`
  (image ID atteso
  `sha256:5e32aaa1bbe3559e81db7706ed4286248f18d27cfdb186f6b851bf786eb43075`).
  L'immagine dichiara vLLM FujitsuPolycom
  `e02b174693e13859de61811b5e8cd13d5308e259` e include SparkCache
  `66057174301a4759ca3a45207ea41016689449cb`. La cattura locale dell'OCI contiene 2.502
  file Python, con 2.301 file comuni al checkout e02, l'unica differenza comune in
  `distributed/kv_transfer/kv_connector/v1/metrics.py` e tutti i 17 file critici uguali
  byte per byte. Il METADATA installato riporta invece la versione obsoleta
  `0.1.dev1+gd377796e8`: viene registrata separatamente e non sostituisce il pin della
  composizione verificata. Sorgenti e config del registry hanno chiuso i controlli statici
  di API, geometria ed entrypoint; identità installata sui quattro rank, gate numerico
  SIRCL e caricamento effettivo restano prove runtime obbligatorie.
- **Confini:** il primo candidato usa il nuovo motore con SparkCache disattivata, mantenendo
  dal runtime E07a checkpoint target FP8
  `690b705278a3a58e538fcb37c2ca8b5f9511213c`, DFlash2
  `bf582e4eacc1810f76656d1811693ff6c6737d2a`, TP4/DCP1, KV cache 16 GiB
  `fp8_e4m3`, block size 2304, adaptive k3/k5 e SIRCL single-rail sync-prefill. Non sono
  ammesse sostituzioni NVFP4, la ricetta packaged dual-rail/fused SIRCL o modifiche ai
  pesi. Il launcher pubblico R10 con NVFP4 e block 256/512 non è una ricetta candidata.
  R10 richiede però `--attention-backend B12X` per lo sparse MLA del target con head size
  512; il backend KDA resta a selezione nativa, il block resta 2304 e il percorso PCIe
  all-reduce B12X resta disattivato per non interferire con SIRCL. Anche il nuovo default
  all-reduce FlashInfer di R10 viene disattivato, preservando il fallback NCCL/SIRCL E07a.
- **Sequenza:** qualificare prima R10/SparkCache-OFF/SIRCL; solo dopo confrontare
  R10/SparkCache-ON/SIRCL a parità di parametri non legati alla cache. Se questo percorso
  passa, provare SparkCache-ON sul percorso NCCL con la stessa immagine, configurazione
  cache e parametri non legati al trasporto. Ogni variante ammessa mantiene il protocollo
  di tre run; F0 resta il confronto storico congelato e non viene rieseguito.
- **Preparazione cache ON:** i manifest canonici letti in sola lettura sui quattro rank
  coincidono byte per byte. Fissano il target a
  `c652a5964160ed137dce29253f7d8a6a1477f17c68a82042b65d94b1f97683f4` e il draft a
  `5d3e05d24de860b98fa172f480a859d808e3022731d0fda1e1a83ad8d776e1ff`.
  La configurazione locale non ancora avviata usa il profilo manager-page
  `glm53-flash-hybrid`, root `/cache/jit/sparkcache-context`, span 4096–262144,
  store/restore attivi e streaming/CUDA restore disattivi. Un override circoscritto del
  connector SparkCache applica il `cache_salt` vLLM a ogni chiave esterna tramite un digest
  SHA-256 con dominio separato; `None` e stringa vuota conservano le chiavi legacy e il
  salt in chiaro non viene trasportato, conservato o registrato. I test CPU e la review
  indipendente restano gate locali; OFF deve qualificarsi prima di qualsiasi avvio ON.
- **Lifecycle:** durante preparazione e ispezione resta in servizio E07a, sperimentale e
  non promosso. La finestra autorizza una transizione runtime coordinata su quattro rank,
  non modifiche host/rete o reboot. Ogni boot richiede `/health` 200 e i gate Rome/tool
  entro 120 secondi; un fallimento impone full-cluster down e rollback esatto al payload e
  overlay E07a dell'archivio immutabile `glm53-e07a-20260912` (manifest
  `626eca7eeb2ec2f39464eb49e8f7d4e409c5b4d4e697e1a9660dcbdd509664fc`). Lo staging
  additivo dell'archivio OCI usa i link DAC esistenti in tre copie parallele da rank 0:
  0→1 e 0→3 dirette, più 0→1→2 con `ProxyCommand` e identità host già fidate. Il percorso
  si arresta su un errore, senza fallback management né modifiche a route, host key o rete.
  Il primo snapshot ha verificato 58 parti su ogni destinazione e trasferito
  4.026.531.840 byte nuovi per rank in 36,973 s. Il secondo passaggio ha distribuito le
  parti residue in 23,584 s e verificato su tutti i rank l'archivio completo di
  9.673.946.624 byte, SHA-256
  `3b16752345aa69b5da91691022ba2adfe5ee9f63474762d38cfe3c22c2b33e53`; il candidato R10
  non era ancora stato avviato a quel punto. Il gate numerico R10/SIRCL successivo ha
  superato 20/20 casi, ma il primo avvio OFF è fallito prima di `/health`: sul rank 3
  DFlash2 non ha trovato un backend attention per la propria firma non causale
  sliding-window (head size 128, BF16, KV `fp8_ds_mla`, block 2304, `use_mla=false`). Il
  caricamento osservato su rank 0 era ancora parziale, 19/62, e non prova un load completo
  sui quattro rank. Rome, tool-call e Rigmark non sono partiti: 0 run e 0 richieste native.
  Il tentativo non è qualificato, SparkCache ON resta bloccata e nessuna correzione del
  drafter viene scelta prima della diagnosi sorgente. La diagnosi ha poi identificato il
  campo R10 `SpeculativeConfig.kv_cache_dtype`, applicato esclusivamente alla cache del
  drafter. L'overlay preparato per il tentativo 2 fissava lì `fp8_e4m3`, come nel
  runtime E07a, preservando `--kv-cache-dtype fp8_e4m3` globale, B12X, block 2304, la
  tabella adattiva k3/k5 e SparkCache OFF. Il dry-run sui quattro rank differisce dal
  tentativo 1 solo per quel campo nel JSON speculativo; BF16 non è selezionato. Anche la
  preparazione SparkCache ON ora eredita esplicitamente quel campo draft-only e viene
  congelata in un manifest separato; il precedente overlay ON privo della correzione non
  è un candidato eseguibile. Il tentativo 2 ha superato il precedente punto DFlash2 sui
  rank 1-3, ma è poi fallito prima di `/health` durante il profiling
  `determine_available_memory`: il receipt rank 0 conserva l'errore RPC propagato
  `cutlass_gemm_caller ... Invalid status`, senza lo stack originale dell'operatore né
  rank worker, layer o shape esatti. Il cluster non ha eseguito Rome, tool-call o Rigmark:
  ancora 0 run e 0 richieste native. Il successivo ripristino E07a ha completato `up` con
  exit 0, raggiunto `/health` 200 e superato Rome più tool-call 2/2. Il controllo finale ha
  verificato un solo stack E07a corretto sui quattro rank, idle 0/0 e 47 °C con flag
  termici e power-brake inattivi; questa misura descrive il runtime ripristinato e non
  assegna la causa del fallimento R10.
- **Diagnostica FP8 e tentativo 3:** con lo stack a quattro rank nuovamente fermo, il probe
  GPU isolato B12X ha completato sette casi senza errori runtime: il percorso block ha
  superato 2/4 soglie numeriche (`M=128/8192`, relative L1 circa 0,00141) e non le ha
  superate per `M=1/4` (circa 0,00287), mentre il percorso online per-tensor Cutlass ha
  superato 3/3. La variante che disabilitava due classi block ha selezionato Marlin e si è
  fermata prima dell'aritmetica, quindi non è diventata una ricetta serving. Il probe con
  `KernelConfig(linear_backend="triton")` ha poi selezionato
  `TritonFp8BlockScaledMMKernel` e superato 7/7 casi: block 4/4 con relative L1
  0,001404–0,001419 e online Cutlass 3/3 con 0,001398–0,001410, picco CUDA 353.374.208
  byte e nessun peso montato. Il receipt è
  `/private/tmp/e09-r10-operations-cxv8a5wx/r10-kernel-probe-linear-triton-summary.json`,
  SHA-256 `02a50dd112fd83a5e92d27ed226fce74bd3ba39cda8fa551c25ee3048c798d84`.
  Il tentativo serving 3 è stato quindi avviato con l'unica delta
  `--linear-backend triton` rispetto al tentativo 2; non usa
  `VLLM_DISABLED_KERNELS` e non cambia immagine, pesi, KV, trasporto o cache. Target e
  draft sono stati caricati su tutti i rank e il profiling che aveva fermato il tentativo
  2 è passato. La creazione delle view KV è poi fallita prima di `/health`: il layout BLHNC
  aveva block stride 14217984 e page size 1179648, quindi il manager block 2304 non poteva
  essere diviso in 36 kernel block da 64. Il runtime ha suggerito block 64 oppure
  `VLLM_KV_CACHE_LAYOUT=LBNHC`, ma il record non seleziona nessuna delle due modifiche e
  non conclude un'incompatibilità generale di R10 o SparkCache. Non sono partiti Rome,
  tool-call o Rigmark: 0 POST funzionali, 0/3 run e 0/162 richieste native. L'`up` posseduto
  è stato interrotto, il full-cluster `down` è passato e il receipt successivo verifica
  zero container, processi GPU e flusher sui quattro rank. Il ripristino E07a ha poi
  completato `up` con exit 0 e `/health` 200. L'operatore ha riferito che il sandbox ha
  negato la rete al primo watcher locale; il receipt associato conserva la richiesta TERM
  sul PID esatto, ma non stderr grezzo né lo stato di uscita osservato. Una riproduzione
  locale successiva, distinta dal watcher originale, conserva `URLError` con errno 1 senza
  escalation del sandbox. Rome e tool-call hanno passato
  2/2 al controllo ripetuto, ma circa 6 minuti e 27 secondi dopo il completamento del
  controller, quindi il requisito dei 120 secondi non è soddisfatto. Non si è osservato un
  fallimento funzionale e non è stato eseguito un restart aggiuntivo. Il controllo finale
  mostra un singolo stack E07a esatto sui quattro rank, restart 0, idle 0/0, 47 °C e tutti
  i flag termici/power-brake inattivi. Un probe CPU successivo sull'immagine R10 ha
  riprodotto con la funzione installata l'esatto errore geometrico BLHNC e superato i due
  controlli allocator in 3,886 s, senza esporre GPU, caricare pesi o modificare il servizio
  E07a sano. Il probe spiega il guard ma non verifica un rimedio supportato e non costituisce
  un quarto boot. SparkCache ON resta non avviata e bloccata; la nuova
  preparazione ON eredita esattamente il backend lineare Triton e differisce dall'OFF
  soltanto per il mount read-only del connector e `--kv-transfer-config`.
- **Preparazione tentativo 4:** il proprietario ha autorizzato una correzione circoscritta
  del layout R10, senza cambiare immagine, pesi, block size, dtype, backend, trasporto o
  stato SparkCache-OFF. Tre override read-only aggiungono un'identità di allocazione con
  default condiviso, conservano target B12X/KDA nell'allocazione esistente e assegnano a
  ciascuno dei cinque layer DFlash una pagina densa separata. Il divisore di memoria è la
  somma tra il pool target condiviso, incluso il padding già applicato da R10, e tutte le
  pagine draft; resta un solo `num_blocks` globale e il totale non supera il budget KV.
  Il test CPU nell'immagine R10 esatta ha usato tre block scheduler e cinque layer draft:
  6 allocazioni, 60.171.264 byte allocati su 60.761.088 disponibili, storage distinti,
  copia multiblock, DCP replicato e percorso target-only invariato. Ha anche riprodotto il
  guard originale 14217984/1179648. E07a ha mantenuto lo stesso container, PID e health
  200 prima e dopo. Il manifest privato della preparazione è
  `cache-fix-attempt4/CACHE-FIX-ARTIFACTS.sha256`, SHA-256
  `e6825e093f5f8ae52b0e5b1de3fea27d990f51d5402da1efea4d1b1069a41571`.
  Il successivo boot serving ha confermato l'effetto del fix: tutti i rank hanno caricato
  target e draft, completato i grafi e creato le view; `/health` ha risposto 200. La nuova
  capacità è 1.435.070 token, 5,47× al contesto 262.144. Il primo gate Rome, però, ha
  restituito HTTP 200 con esattamente 64 caratteri `!` e `finish_reason=length`; il gate
  tool è stato saltato e Rigmark è rimasto a 0/3 run e 0/162 richieste. La ricetta esatta
  del tentativo 4 è quindi scartata e il `down` coordinato è passato. Il ripristino E07a
  ha completato `up` con exit 0 in 971,241 s, raggiunto health 200 e superato Rome più
  tool-call 2/2 in 8,826 s dal primo health. Il controllo finale mostra un solo stack E07a
  esatto per rank, restart 0, idle 0/0, 48 °C e tutti i flag termici/power-brake inattivi.
  Questo non conclude un'incompatibilità generale di R10 o SparkCache, che era OFF;
  SparkCache ON resta non avviata e bloccata.
- **Tentativo 5, primo avvio strumentato fallito:** il proprietario ha autorizzato il backport
  circoscritto della correzione upstream `618562444d73742e4e872defcad4d37f477f5c59`
  per la coda di 0–3 token dopo i pool completi. La revisione del call path ha mostrato che
  DCP1 decode-only usa anche `expand_pool_ids_physical`; la revisione 3 applica quindi la
  stessa formula `live_history=min(2048, complete_pools*4)` al helper logico upstream e al
  helper fisico R10, senza altre modifiche al kernel. Il primo runner preparato è stato
  respinto da Docker per il nome runtime `nvidia` prima di creare un container o eseguire
  casi GPU. Il runner corretto usa l'invocazione già verificata `--gpus all`: sull'immagine
  R10 esatta il modulo originale ha fallito gli 8 casi corti attesi sulle due route, mentre
  la revisione 3 ha superato 16/16 casi, incluso il prompt da 25 token, i confini
  2047/2048/2049/2051 e il remap fisico lungo, in 29,89 s. Anche il replay CUDA isolato
  della diagnostica ha passato il proprio controllo senza pesi in 4,947 s. Il manifest
  privato patch/test è
  `pool-fix-attempt5-rev3/PATCH-TEST-ARTIFACTS.sha256`, SHA-256
  `206d2c462c38cca87ce9ee3e8576a53500ee925cc5803d7ce9536c26a860dade`; il
  candidate strumentato è fissato da manifest SHA-256
  `21ddf661ced69099f9bcd5a0b66e964d12a80312fd0a88e43047523584ac324d`.
  Lo staging, i dry-run, gli hash e lo stato fermo dei quattro rank sono passati. Il primo
  boot OFF strumentato è però fallito durante il caricamento: il wrapper diagnostico non ha
  esposto al controllo di `model_runner` il modello GLM5 strumentato e i log hanno riportato
  `E09 numeric diagnostics require the instrumented GLM5 model`. Il controller è stato
  fermato con exit 143 dopo 457,45 s e la verifica successiva ha trovato zero container,
  GPU idle e flusher inattivo su tutti i rank. Non sono stati raggiunti health, gate,
  inferenza o Rigmark. Questo fallimento riguarda l'involucro diagnostico e non dimostra un
  fallimento del fix pooled-tail. Un secondo boot con il wrapper corretto ha caricato target
  e draft su tutti i rank, completato `up` con exit 0 in 1002,05 s e raggiunto health alle
  11:50:39 UTC, ma il primo gate ha restituito HTTP 200 con 64 `!` e
  `finish_reason=length`; il tool gate è stato saltato. I 16 record diagnostici appartengono
  ai quattro step di warmup per rank, tutti precedenti a health: il budget era già esaurito e
  la richiesta Rome fallita non è stata osservata. Gli output DFlash non descrivono i layer
  ausiliari target; la causa resta irrisolta. Il `down` coordinato è passato in 12,973 s.
  Nessun boot pulito OFF, run Rigmark o richiesta nativa è iniziato. Il recipe OFF
  strumentato esatto è scartato, senza generalizzare il risultato a R10, al fix pooled-tail
  o a SparkCache. SparkCache ON è stata preparata e depositata sui quattro nodi, senza
  avviarla. Il ripristino E07a ha completato `up` con exit 0 in 971,111 s: health 200
  alle 12:10:13 UTC e gate Rome/tool-call 2/2 in 8,801 s. La verifica finale conferma
  un solo stack esatto per rank, restart 0, ambiente SIRCL corrispondente, idle 0/0,
  due sole richieste di gate, flusher spenti e 49/49/48/49 °C senza flag termici o
  power-brake attivi. Il controllo locale iniziale dei flusher accettava solo exit 3:
  exit 4 con `inactive` era invece atteso per le unità transitorie già raccolte;
  nessuna riparazione dei nodi è stata necessaria. Il proprietario ha richiesto di
  proseguire la root cause analysis direttamente nel contesto root, senza delegarla.
  L'[analisi diretta](#e09-root-cause-analysis) distingue i blocchi già risolti
  dal guasto di generazione ancora non localizzato. Il nuovo probe privato lega i quattro
  campioni alla richiesta Rome effettiva e segue input, layer, logits e token validi:
  catena request-ID e otto controlli numerici CPU passati, senza inizializzare CUDA o
  alterare E07a. Replay GPU, staging e nuovo boot restano non eseguiti. Il difetto di
  lifetime SharedExperts descritto dalla PR 706 è una pista concreta nel sorgente R10,
  senza riproduzione SM121 né prova causale per i 64 `!`.

## E09 root cause analysis

The investigated failure is **incorrect generation after successful startup**: R10
returned HTTP 200 with 64 exclamation marks. A named-request trace now places the
numeric divergence in the first decoder block, and an isolated GPU test reproduces
an incorrect FP8 selection for BF16 KDA projections. The clean corrected OFF process
now answers the Rome request correctly, but exposed a second defect in pure-decode
sparse-attention selection. Its physical cache slots were translated twice. A GPU
reproducer confirms the narrow provider fix. The corrected four-rank SparkCache ON
boot passes both functional gates and all three native Rigmark runs have completed
on Beast0 with the same process. SparkCache performs real external restores. The
combined recipe is **unresolved / decision_required**, without promotion: C2 is
consistently below F0, while the small code/C1 gains and faster prefill do not settle
that primary-workload tradeoff.

The owner requested that this investigation remain with the root agent. This record
separates observed failures, source findings and untested hypotheses.

### What the evidence establishes

| Finding | Evidence | Limit |
| --- | --- | --- |
| The engine started | Four-rank weight-load logs, graph capture, KV views, health 200 | Readiness does not prove correct generation or GPU weight contents |
| The drafter alone does not explain the response | Attempt 4 rejected all 205 proposed draft tokens, yet returned repeated exclamation marks | The target calculation, verifier/sampler and output handoff still need tracing |
| The previous trace missed the failed request | All 16 target records preceded health and the API POST; four startup steps per rank exhausted the budget | Zero warmup outputs cannot identify the operator responsible for the real request |
| Critical checkpoint samples are populated | Read-only rank-0 sampling of embedding, output head, final norm and last-layer attention/FFN mHC parameters: nine categories, 2,136 data bytes; samples finite and nonzero | Samples are not a full checkpoint validation or inspection of GPU-loaded tensors |
| The checkpoint names match the loader mapping | Exact R10 `glm4_1v.py` wrapper maps `model.language_model.` to `language_model.model.` and the output head separately | Name agreement does not prove every parameter was loaded correctly |
| SparkCache ON runs and restores state | Three native runs, 162 requests, 45/45 basic output gates; 21/27 native replay requests restored externally on all four ranks | This measures the combined R10/SIRCL/SparkCache recipe; short replay equality is not a long-continuation semantic test |

The old-vLLM SparkCache port remains partial. R10 was selected to use its integrated
connector. The clean OFF boot corrected the numeric collapse but failed its tool gate;
the subsequent pure-decode selection fix passes the ON boot's two functional gates.
E07a is the old working
engine plus experimental single-rail SIRCL sync-prefill over the existing DAC ring;
the full SparkRing mesh was not installed.

### Earlier blockers and their disposition

The draft cache inherited the target-specific `fp8_ds_mla` type. The explicit
draft-only `fp8_e4m3` field passed that initialization point. A later profiling error
reported `cutlass_gemm_caller ... Invalid status`; its exact operator stack was not
retained. Selecting Triton linear kernels passed isolated numeric cases and the next
serving boot passed profiling, without proving the missing original stack.

The target/KDA shared allocation could not give dense DFlash layers a valid BLHNC
view when a manager block of 2304 split into 36 kernel blocks of 64. The separate
draft-allocation override passed the exact-image CPU allocator checks and serving
proceeded past KV view creation.

The pooled-tail indexing defect was reproduced independently: eight short-tail
cases failed in the original source, and the corrected logical/physical helpers
passed all 16 GPU cases. Attempt 5 still returned the same invalid content after
that fix. It fixes the demonstrated index calculation, not the whole serving failure.

The first instrumented boot also exposed a wrapper omission in our diagnostic code.
Forwarding the three diagnostic methods fixed that startup guard. The subsequent
trace still had a separate request-selection defect, described below.

### Source findings from the direct investigation

The comparison uses R10 composition
`e02b174693e13859de61811b5e8cd13d5308e259`, image ID
`sha256:5e32aaa1bbe3559e81db7706ed4286248f18d27cfdb186f6b851bf786eb43075`,
against the working vLLM revision
`487ecf187d3dfe74d2cf6119a92881dba403c219`. B12X Python source was extracted from the
pinned local image layer, not inferred from the latest branch.

The input preparation and graph replay paths use persistent embedding buffers.
Inspection has not established a wrong buffer address. Likewise, reading the B12X
mHC pre/post implementations and matching their supported shapes has not established
a terminal-layer mHC defect. The zero warmup trace does not justify that attribution.

There is a concrete buffer-lifetime defect to keep in scope: this R10 source's
`model_executor/layers/fused_moe/runner/shared_experts.py` waits for the producer
stream but omits recording the output on its consumer stream. The fork's
[PR 706](https://github.com/local-inference-lab/vllm/pull/706) supplies that ownership
correction and an isolated CUDA reproducer. Its published test demonstrates premature
storage reuse. This is a plausible applicable defect, not proof that it caused our
64-character response; the reproducer has not run on our SM121 nodes and the patch
has not been applied. Instrumentation that synchronizes tensors could also mask a
stream-lifetime race, so a diagnostic PASS alone would not qualify a clean candidate.

Healthy E07a on the same fabric weakens a basic link-failure explanation. It does not
exclude an R10-specific collective integration error. No network setting has been
changed during this analysis.

### Request-bound diagnostic prepared locally

R10's `v1/worker/gpu/warmup.py` creates `_warmup_...` requests and runs the normal
execution path. The previous `not dummy_run` condition therefore admitted them.
The replacement reserves four steps for the exact named Rome request and accepts
the eight-hex internal suffix added by `InputProcessor.assign_request_id`.

The private `root-rca/diagnostics-v3/` package contains the complete OFF overlay,
four module overrides, the named API body, tests and the interpretation procedure.
It observes prepared embeddings before forward, per-layer hidden/residual/mHC state,
actual target auxiliary states, final normalization, selected hidden states, logits
and sampler counts. A token slot is read only when `num_sampled > 0`.
Empty rank-local logits are handled explicitly.

Stage reductions inspect at most 32 rows to bound warmup/capture cost. The runner
rejects a named diagnostic batch above that limit; the unchanged Rome gate previously
used 25 prompt tokens. The process is diagnostic-only and cannot produce Rigmark
performance measurements.

Completed verification:

- Exact-source API-to-engine request-ID chain: two header/body cases pass; 56 warmup
  or unrelated batches excluded; the four-step budget remains intact.
- Eight CPU numeric checks pass, including padding, nonfinite values, zero counts,
  final-layer indexing, reset, bounded rows and empty logits.
- The CPU test used a separate process in the existing E07a container, with CUDA hidden
  and never initialized. Container ID, process, start time, restart count and health
  remained unchanged.
- Four Python overrides parse and the private overlay passes shell syntax validation.
- Offline launcher rendering passes for all four ranks: the only argv changes are
  the four diagnostic source paths and the named-request environment entry.

The owner subsequently authorized this diagnostic window. The new recorder passed
10/10 CUDA checks on Beast0 without model weights, followed by live staging and
render checks on all four ranks. The instrumented R10 boot completed in 1032.433
seconds and reached health at 13:29:40.982647 UTC on 2026-09-13. The first named
request again returned exactly 64 exclamation marks; the gate failed within 4.625
seconds of health. The tool request was skipped. Actual counts: one coherent
request, one functional failure, zero tool requests and zero native Rigmark runs
or requests. Coordinated R10 down completed in 13.218 seconds.

### Interpretation procedure used for the named request

Use one coordinated diagnostic window and one named coherent-response request on
R10 OFF. Check the new recorder's weight-free CUDA replay before loading weights.
Follow the first divergence through the recorded boundaries:

| Observation on the actual named request | Next focused check |
| --- | --- |
| Prepared embedding is nonzero but graph input differs | Persistent buffers, padding and capture/replay binding |
| A layer first loses valid state or produces nonfinite values | That layer's attention/MoE/mHC operations and collective output |
| Raw hidden output is zero but residual/mHC state is live | Materialized residual and auxiliary output; raw hidden alone is insufficient |
| Selected hidden state is valid but logits collapse | Loaded output-head contents and projection/gather |
| Logits are valid but emitted token disagrees with the applicable sampling rule | Greedy/rejection sampler and asynchronous output handoff |
| The failure disappears under instrumentation | Stream lifetime and synchronization sensitivity, including PR 706; then a clean verification |

Record the actual request ID on every diagnostic boundary. Missing named-request
records mean the probe failed; never substitute warmup. A diagnostic success requires
a separate clean process before OFF qualification or any SparkCache benchmark.

### Actual request trace and reproduced quantization defect

The request ID was `chatcmpl-e09-root-rca-20260913-a588eacb`. All 16 target records
belong to that request after health: four steps on each rank, with no warmup or
unattributed records and no missing required boundaries. Step 1 used PIECEWISE with
25 tokens; steps 2–4 used FULL with six tokens. The first-step embedding has normal
finite values (maximum absolute value about 0.0584), but the first block's residual
already reaches about `4.31e29`. Its raw hidden output is zero. The last block
materializes a nonzero residual of about `2.53e29`; final normalization produces all
zeros, as do the actual selected hidden states and logits. Each recorded sampler
step selects token 0. The four ranks agree on these extrema.

These huge tensors remain finite. The recorder's `rms: null` is an overflow of its
FP32 sum of squares, not evidence that those tensor elements contain NaN or Inf.
Their scale is also consistent with overflow in the final RMS normalization. The
terminal mHC operation does materialize the residual; the earlier speculation that
it simply returned zero is not supported.

Docker inserted repeated timestamps at 16-KiB transport fragment boundaries inside
the long JSON records. The offline decoder removes only an exact repetition of the
record's timestamp after validating its original byte offset is a positive multiple
of 16384. It reconstructed eight such insertions per rank, retained raw-log hashes,
and parsed all named records without rerunning inference.

Direct comparison then identified the precision-selection error:

- The checkpoint's quantization exclusions name text layers as `model.layers.*`.
  Its actual tensor names instead use `model.language_model.layers.*`.
- R10's multimodal mapper translates the tensor namespace to
  `language_model.model.layers.*`, but did not translate the exclusion namespace.
  All five KDA projection selections therefore missed the exact exclusion match.
- The new shared KDA implementation passes through the FP8 configuration. The
  earlier working GLM adapter explicitly constructed these projections in BF16.
- R10 consequently allocates serialized FP8 weights and scale parameters for
  checkpoint tensors that are BF16 and have no serialized scales. The unused scale
  retains its initialization value, `-3.4028234663852886e38`.

A GPU test of the real layer-0 `g_b_proj` weights reproduced this fault in the exact
R10 image: the incorrect path produced finite values up to `1.0819e38`, while the
BF16 reference reached only 0.29296875. The BF16 control matched the reference
exactly. Separately, six mHC pre/post-pre cases with real first-layer parameters
passed, including 25-, six- and one-token shapes and caller-owned versus functional
outputs. These tests require only small checkpoint portions, not a full serving load.

The candidate adds one alias to `Glm5NextForConditionalGeneration.hf_to_vllm_mapper`:
`model.layers.` → `language_model.model.layers.`. It preserves the parent mapper and
does not disable FP8 for the model. Verification of the actual patched module found:

- 170/170 KDA projection selections correctly excluded across all 34 KDA layers;
- all 442 KDA checkpoint weight tensors BF16; three initial dense MLPs still FP8;
- all 76,108 native checkpoint weight-name mappings unchanged and the exclusion
  mapping idempotent;
- eight GPU projection cases matching the independent BF16 reference exactly,
  including all four TP shard geometries and the replicated gate shard.

The clean module SHA-256 is
`7ae82a961e505fd5ad1c2c289ea1338a15be94317082c0d02e02f1017e7fe47f`.
The private overlay `e09-r10-off-quant-map-fix.env` has SHA-256
`44f73ea265467e75a353e3a096a482c15c9ac780ed4da17945831d06b21ed657`;
its artifact manifest SHA-256 is
`d7d9e95ee462e502a70b3267f36c14cc30becaa3a9d714c4919c5c5d86edb785`.
Four-rank staging and rendering passed. Relative to the diagnostic boot, it removes
both diagnostic environment entries and all four diagnostic mounts, then adds only
the clean corrected model module. The image, weights, draft-FP8 field, Triton linear
backend, cache-allocation and pooled-tail corrections, adaptive k3/k5 and E07a SIRCL
transport remain identical. No instrumented forward runs in this candidate.

### Clean OFF serving and the second decode defect

The clean mapper-corrected OFF boot completed `up` with exit 0 in 726.890 seconds
and reached health at 14:15:58.506493 UTC on 2026-09-13. It loaded 77.54 GiB of
target weights in 535.713479 seconds and provided 1,435,070 KV tokens. The Rome
gate returned “The capital of Italy is Rome.” with eight completion tokens and a
normal stop. The tool gate still failed: the generated function name did not match
the declared `get_weather`. Initial gates passed 1/2, with the failure recorded
9.237 seconds after health. A later temperature-zero request finished 120.800
seconds after health and is a diagnostic follow-up, not a passing timed gate.

The owner-authorized continuous diagnosis retained that process. Its total was
**nine inference requests, all HTTP 200**: Rome 1/1 correct; tool requests 1/6
correct and 5/6 incorrect; two other completions used intentional diagnostic token
caps. There were zero native Rigmark requests. The five tool failures must not be
hidden behind successful transport or the later mixed-prefill success.

Re-prefilling the exact tool prefix predicted the correct `get` token, while the
earlier continued decode generated a different function name. A fresh cache salt
did not correct it. The same thinking-off tool request succeeded when issued
alongside a 65,536-token prefill companion: `get_weather` with city `Milan`, twelve
completion tokens. The tool and companion took 17.835 and 27.040 seconds. This
isolates a difference between pure decode and mixed prefill; it does not by itself
prove a CUDA graph defect.

The installed R10 source explains that difference:

- `models/glm5next/nvidia/pooled_indexer.py` already calls
  `expand_pool_ids_physical` for DCP1 pure decode and returns physical token slots.
- It omits `get_b12x_physical_selection`, so the runtime-checkable
  `B12xPhysicalSelectionProvider` protocol in `b12x_mla_sparse.py` does not match.
- `B12xMLASparseImpl.forward_mqa` therefore takes its logical-index fallback and
  translates those physical slots again. Mixed prefill produces logical slots and
  correctly needs that conversion once.

The private fix adds the missing provider method, returning views of the existing
slot/count buffers only for DCP1 pure decode. It returns `None` for mixed prefill
and DCP greater than one. It adds no allocation, GPU copy or kernel and preserves
the existing speculative-snapshot methods. The corrected module SHA-256 is
`a8b55ac2a80b94b263160a5a17bd88fbf0663cdc536a3a0c4447b390ee28dfe9`.

The GPU reproducer executes the actual patched indexer class and backend dispatch,
with the workspace and final attention computation stubbed. Actual pool expansion
and the old fallback conversion run on the GPU. For positions
`24, 160, 163, 2303, 2310, 4608`, the original path has respectively
`25, 161, 164, 2048, 2048, 2049` wrong selected slots. The patched path has **zero
wrong slots in all six cases**; buffer aliasing, unchanged mixed-prefill mapping
and the DCP2 fallback also pass. The probe took 6.279 seconds and allocated at most
20,037,120 bytes. This proves dispatch and addresses, not full-model generation.

A separate actual B12X KDA decode probe matched an independent PyTorch reference
for three accepted-token lengths (1/3/6), with zero output error and state error
below `3e-8`. No KDA kernel patch was applied. The first attempt to create that
small secondary CUDA context on rank 0 failed from insufficient available memory
before any cases ran; it was rerun successfully on rank 3. The serving process
remained healthy and was not reloaded for these operator probes.

### Direct corrected SparkCache ON window

The owner requested going directly to three native Rigmark runs with SparkCache
if generation works. To avoid another OFF load followed by an ON load, the next
boot combines the proven provider fix with the already prepared SparkCache
connector. The short Rome/tool gates are below the connector's 4096-token minimum
span. No OFF benchmark or intermediate F0/E07a reload is planned.

The active candidate overlay is
`scripts/node/etc/local/e09-r10-on-physical-selection-fix.env`, SHA-256
`5a63029a1ed99ff3fd5ee161ac184393f0dc1e6b756c652aa1961f1575ecb306`, with package
manifest SHA-256 `85671185c292c694c94f1cef3e2cc9dddf3537e174067add9f3cfb9b0c4b774b`.
It retains the clean mapper, separate allocation and pooled-tail fixes. Its
launcher SHA-256 is `d85801569b20f4f7def294bf6781665a5064804a1b4de0141d6d984fb58447d7`.
The connector SHA-256 is `3f44e92ae8f6e384b73a0cde94fd234e1d7b6017b0d9b493b187f5e1f8f74b63`
and its configuration SHA-256 is
`c5063709faa90b25d7196909e59832ede9805f497a4829c7f6f0f079eb894b16`.

Staging and rendering passed on all four nodes: ten source hashes per rank, no
numeric diagnostics, and only the provider mount added to the previously prepared
ON command. Coordinated OFF down passed in 14.436 seconds; the eight-direction
fabric check passed. The ON boot completed `up` with exit 0 in 758.013 seconds,
reaching health at 15:07:47.100277 UTC. Both functional gates passed within 3.073
seconds: Rome returned eight tokens and a normal stop; the tool request returned
`get_weather` with city `Milan`, 42 completion tokens and `finish_reason=tool_calls`.
These are two successful gate requests, separate from native benchmark counts.
Runtime inspection at
15:04:32 UTC confirms all four expected containers, restart counts zero, matching
environment and mounts, SparkCache enabled and no foreign GPU process. Temperatures
were 49/47/46/47 °C with thermal and hardware power-brake flags inactive.

Rigmark remains native protocol 1.1.0, revision
`c671b52a97f3cc01919c18d8d1f8e4f01243290c`. Its source and prompt fingerprints match
the frozen comparison. Six executable bits had changed since staging; restoring
their original modes recovered the exact saved whole-worktree fingerprint without
changing file contents. Three commands are prepared on Beast0 with distinct cache
salts, 54 inference requests each, the original 8192-token decode, 8K/32K/64K
prefill and 256-token code C1/C2/C4 settings. The same ON process must span all three
runs. The completed results and actual connector evidence follow. Immediate native
replay alone does not establish a SparkCache external-restore benefit; the worker
restore logs supply that evidence here.

### Completed three-run SparkCache result

All three native runs completed on 2026-09-13 with one corrected ON weight load and
the same four container IDs, without an intervening reference reload. Native run
windows were 15:08:32–15:20:37, 15:21:07–15:32:36 and 15:33:19–15:44:31 UTC.
Every source, prompt and benchmark-setting fingerprint matches the frozen F0
protocol; each run uses a distinct cache salt. F0 was not rerun.

| Metric | Fixed F0 median | Median of three native ON runs |
| --- | ---: | ---: |
| Long-code decode, tokens/s | 50.401 | 50.755 |
| Code C1 aggregate, tokens/s | 37.223 | 38.570 |
| Code C2 aggregate, tokens/s | 57.009 | 50.682 |
| Code C4 aggregate, tokens/s | 71.084 | 66.235 |
| Prose decode, tokens/s | 29.220 | 29.875 |
| Code TTFT, seconds | 0.410 | 0.397 |
| Prose TTFT, seconds | 0.397 | 0.383 |
| C1 per-stream TTFT, seconds | 0.404 | 0.349 |
| C2 per-stream TTFT, seconds | 0.616 | 0.468 |
| C4 per-stream TTFT, seconds | 0.935 | 0.661 |
| 8K cold prefill, tokens/s | 2112.218 | 2378.337 |
| 8K replay prefill, tokens/s | 4649.988 | 9625.666 |
| 32K cold prefill, tokens/s | 2201.970 | 2554.465 |
| 32K replay prefill, tokens/s | 21524.192 | 34437.170 |
| 64K cold prefill, tokens/s | 2201.282 | 2302.833 |
| 64K replay prefill, tokens/s | 35971.377 | 37504.070 |

Counts are explicit: **3/3 native runs, 162/162 requests, zero native validation
findings and zero stream errors**. Code, prose and structured basic output gates
each pass 15/15, or 45/45 combined. Long-code length caps are 0/15. Concurrent code
retains the intentional 256-token cap per agent; those requests do not test long
parallel completions. The 27 native cold/replay pairs produce equal eight-token
continuations in 27/27 cases, including replay requests without an external hit.
These are native basic gates, not execution tests of the generated code.

Each node recorded 27 snapshots and 27 durable commits with no failed publication
or error log in the native windows. External restoration occurred on all four ranks
for **7/9, 8/9 and 6/9** native replay requests, or **21/27** overall. The other six
replays remain in the measurements. In particular, the three 32K replay medians
were **38065.721, 34437.170 and 2360.098 tokens/s**: the favorable aggregate median
must be read alongside immediate replays taking about 12–14 seconds. The cause of
those missed immediate restores has not been isolated by a controlled test.

The repeated C2 results, 48.965/50.994/50.682 tokens/s, are below all saved F0
per-run C2 medians, 54.515/57.009/57.705. Long-code gains are small and overlap
observed variation. C4's 63.597/66.235/66.719 is below the fixed F0 median but lies
inside F0's broad 55.757–75.740 range, so it is not established as a disqualifying
C4 regression beyond that noise. No early stop was warranted on that evidence;
the authorized three-run series completed. Faster prefill and measured TTFT do not
silently compensate for the C2 cost. The result remains **unresolved /
decision_required**, with a recommendation against promotion for the primary
concurrent-code workload.

There is a comparison limitation: the saved F0 clients used Python 3.11.6 on
macOS, while the owner requested these runs on Beast0, using Python 3.12.3 on
Linux and loopback HTTP. TTFT and end-to-end differences therefore include client
and transport effects. These measurements compare the whole R10/SIRCL/SparkCache
combination and do not isolate the incremental performance effect of SparkCache.

Final verification at 15:45:08 UTC found health 200, idle 0/0, four unchanged
container IDs, restart counts zero, no foreign GPU process, all ten source hashes
per rank unchanged and all flushers inactive. Success counters were 2/56/110 before
the native runs and 164 afterward: 162 native requests plus two successful boot
gates. The API log agrees: 110 chat POSTs and 54 completion POSTs, all HTTP 200,
with no extra inference POST observed.

Thermal samples during prefill and C4 had a maximum of 78 °C and no active software
thermal, hardware thermal or hardware power-brake flag. The final idle readings
were 62/61/61/62 °C, also without those flags. The run-1 sample labelled C4 was
already after that run finished; the run-2 C4 sample had four active requests.
These are sampled observations, not a continuous thermal trace.

Native receipt SHA-256 values are
`8af4b48e704580305f9ec8d787c4ac9f32fe84aff0308cf78c7311fec279deef`,
`a726edff7849fb34ef9b55693f18b9fb8236471ff662671be765c8eecdf9fcce` and
`c805eb50b05348db3ee64ba42ce550f7d62cdc3968319ed6802350bcb0cf0fe8`.
Commands, metadata, raw native receipts, diagnostic failures, final logs and the
verified controller/launcher/module payload are retained in the private archive
`glm53-e09-r10-sparkcache-20260913`. The healthy corrected process remains running
experimentally under the owner's continuous-window instruction; no automatic
reference reload or production promotion was performed.

### Offline diagnosis of replay misses and concurrent decode

The owner approved an archive-only investigation after reviewing the three ON
runs. It added **zero inference requests, model loads, restarts or deployments**.
The original benchmark archive remains unchanged; all 447 manifest entries and
the original frozen F0 receipt hashes passed verification. The separate private
archive `glm53-e09-offline-rca-20260913` contains the analysis programs, source
hashes, 27 cold/replay pairs, 81 concurrent rounds across F0/E07a/ON, CPU evidence
and the detailed report. These are offline evidence tools, not a new benchmark
interface. Production defaults and public APIs are unchanged.

**Replay admission:** every one of the six misses has a last commit log entry
after the replay HTTP header; all 21 successful replays have every rank's commit
entry before it. Recorded late offsets span 6.8–112.4 ms. Two cases are on rank 0,
sharing the API clock; the other offsets compare uncalibrated host clocks.
Uvicorn headers are not exact scheduler-admission timestamps, and the receipts
omit per-request absolute timestamps and request IDs. Pair association uses the
verified native order, unique logged digest prefixes and matching all-rank spans.
The scheduler's actual quorum mask at each native admission was not recorded.

The installed connector commits CPU snapshots in a background thread and adds
their digests to each worker's local inventory. The scheduler learns availability
through worker reports around execution steps. `get_num_new_matched_tokens`
returns an immediate ordinary miss when any physical rank is absent from its
quorum; a later report does not undo that first recomputation decision. Durable
data and scheduler-visible availability are separate events.

A CPU replay of the exact installed connector reproduced this behavior at all
three prompt depths: four completion callbacks but only three delivered reports
produce `(0, False)`; delivery of the fourth report admits 6912/32256/64512 tokens.
The fixture uses TP4/DCP1, synthetic HMA metadata, manager block 2304, the profile's
256-token digest chunks and minimum span 4096. Short-prompt bypass and salt
isolation also passed. Four existing CPU tests passed, including real background
disk publication and quorum admission. This proves the mechanism, not the missing
native admission state. Asynchronous publication/quorum timing is the strongly
supported explanation; no native payload-corruption failure was observed.

**Concurrent decode:** all four logs show zero snapshot, commit or restore events
during every C1/C2/C4 phase. Their 130–136-token prompts are below the publication
threshold; the CPU control creates no store plan or candidate-prefix hashes.
Native counter deltas show zero preemptions. Five inference-time JIT warnings per
rank occur in run-1 C2, but none in C2/C4 of runs 2 and 3, which remain slower.
Thus neither cache payload writes nor initial JIT explains the persistent result.
These findings do not eliminate connector metadata work or unobserved graph costs.

Complete speculative-log intervals contained within the concurrent phases show:

| Phase | E07a accepted/drafted | Corrected ON accepted/drafted |
| --- | ---: | ---: |
| C1 | 792/1446 (54.77%) | 787/1431 (55.00%) |
| C2 | 2229/4230 (52.70%) | 1968/3916 (50.26%) |
| C4 | 5681/10309 (55.11%) | 5215/10439 (49.96%) |

These are partial observations: three C1 and six C2 intervals per configuration,
13 E07a C4 intervals and 12 ON C4 intervals. E07a phase boundaries use log-line
order because its retained post-run tails lack HTTP timestamps. No per-request
acceptance or complete CUDA-graph dispatch trace exists. Lower acceptance supports
a contribution to the slowdown, but does not attribute it to SparkCache or measure
the whole cost. Client/platform and engine differences remain explicit.

The R10 executor's connector aggregator also changes execute/sample reply routing
from one output rank to all four ranks, including steps with empty cache metadata.
A CPU replay of the extracted executor method passed four routing cases. It
demonstrates the additional coordination path, not its serving latency. The
relative costs of that path and the corrected engine remain unresolved; no
shortcut that drops required rank reports has been applied.

The proposed cache remedy is bounded progress for known pending publications and
prompt all-rank availability reporting, preserving four-rank admission and
recompute on failure. Adding a sleep to Rigmark is not a performance fix. A
separate, unexecuted 21-request diagnostic design compares immediate 32K replay,
replay after one second, and replay at that same deadline after a short intervening
completion. Exact attribution would additionally require targeted admission-state
evidence. Isolating decode cost requires a matched control on the same corrected
R10 engine and Beast0 client; prior failing OFF boots are not valid controls.
No follow-up probe, instrumentation, control boot or benchmark was started during
that offline investigation. The subsequently authorized timing probes follow.

### Replay-timing diagnosis stopped at the output gate

The owner authorized 21 diagnostic requests on the already loaded corrected ON
process, directly from Beast0, without subagents, connector changes, an OFF control
or promotion. The planned order was **A–B–C, B–C–A, C–A–B**: immediate replay (A),
replay one second after cold completion (B), and replay at that same deadline with
a short one-token completion starting at 250 ms (C). Each pair had its own nonce
and salt; bridge salts were independent. All tokenizations preceded inference.

Seven offline checks passed for request count, salt isolation, native client use,
deadline handling including a retained bridge overrun, and stopping on HTTP,
stream, token-count or output failures. The live precheck matched the saved four
process identities, ON configuration and source hashes, with health 200, idle 0/0
and success counter 164. The probes used the unchanged Rigmark 1.1.0 client,
prompt generator and prefill payload: 32,768 input tokens, eight continuation
tokens, temperature zero and `ignore_eos=true`.

The first pair ran at **18:34:58–18:35:26 UTC on 2026-09-13** and triggered the
mandatory stop: its continuations differ by one ASCII space after the period.
Both requests returned HTTP 200, correct 32,768/8 token counts,
`finish_reason=length` and complete `[DONE]` streams. Continuation equality is
**0/1 pairs**; HTTP, incomplete-stream and token-count failures are each **0/2**.
Exactly **2/21 inference requests** ran; the other **19 requests were skipped**,
without replacement attempts. The identical request bodies were verified by hash.

| Pair | Block | Condition | Cold TTFT, s | Replay TTFT, s | Actual cold→replay interval, ms | Result |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | 1 | A | 13.086551 | 14.529793 | 1.706682 | External miss; output differs |
| 2 | 1 | B | — | — | — | Not run after required stop |
| 3 | 1 | C | — | — | — | Not run after required stop |
| 4 | 2 | B | — | — | — | Not run after required stop |
| 5 | 2 | C | — | — | — | Not run after required stop |
| 6 | 2 | A | — | — | — | Not run after required stop |
| 7 | 3 | C | — | — | — | Not run after required stop |
| 8 | 3 | A | — | — | — | Not run after required stop |
| 9 | 3 | B | — | — | — | Not run after required stop |

TTFT uses native Rigmark timing; the interval uses Beast0 monotonic timestamps
from cold response closure to replay HTTP invocation. External hits are **A 0/1
observed, with one of three planned pairs executed**; B and C are unmeasured,
each with zero of three pairs executed. There is no three-value median, measured
bridge deadline or A/B/C comparison. This diagnostic does not replace native
benchmarks or change frozen F0.

All four ranks logged one 32,256-token snapshot and one commit for the unique
digest prefix `2019985afe99`; none logged an external hit or restore. On Beast0's
own clock, its commit log appeared **642.283 ms after replay HTTP invocation**.
This supports publication timing as relevant to the reproduced miss, but log
emission does not expose actual durable-completion time, scheduler admission or
the scheduler's quorum mask. Other rank clocks were not calibrated. With B and C
unmeasured, the diagnosis remains **inconclusive**. No external restore was observed,
so the output divergence cannot be attributed to external-cache payload corruption
from these receipts; its cause is also unresolved.

The correction proposal remains a design, not a deployed fix: distinguish a known
pending publication from an ordinary absent prefix; wait only for that publication
with a bounded monotonic deadline and bounded pending state; notify and wake the
scheduler promptly on worker completion or failure, including while idle. Admit
restore only after **4/4 physical ranks** confirm the same salted digest and
publication generation. On failure, inconsistent generation, timeout or restore
error, retire the pending attempt and recompute normally. Do not block unrelated
requests, repeatedly wait on a failed generation or replace this mechanism with a
one-second benchmark sleep. Qualification needs targeted request/digest traces of
commit, report emission/receipt, admission quorum and fallback, plus resolution or
owner adjudication of the exact-continuation failure.

The private archive `glm53-e09-replay-timing-20260913` retains commands, all prepared
payloads, raw absolute/monotonic request and SSE receipts, output hashes, complete
four-rank logs, association by digest and request order, local checks and the full
proposal. The original archive's 447 manifest entries remain unchanged. API logs
account for exactly 21 preparatory `/tokenize` POSTs and two inference POSTs, all
HTTP 200; no foreign inference was observed. Final verification passed with
counter **166**, health 200, idle 0/0, unchanged processes and source hashes on
all four ranks, zero restarts and inactive flushers. The process remains loaded;
no model load, service restart, public API change, production promotion or automatic
F0 restoration occurred.

### Three cold controls and a local pending-publication candidate

After the stopped timing pair, the owner authorized three cold-only controls and
local preparation of a connector correction. The unchanged native Rigmark prefill
client ran on Beast0 at **19:20:54–19:21:36 UTC on 2026-09-13**, using the exact
32,768 token IDs and nonce from the failed pair, a fresh independent salt for each
request, eight continuation tokens, temperature zero and ignore_eos=true.
The bodies differed only by salt; no new tokenization or replay requests ran.
Five offline preparation checks passed before the three requests.

| Cold control | TTFT, s | Input/output tokens | Exact output matches original cold |
| --- | ---: | --- | --- |
| 1 | 13.447624 | 32768/8 | Yes |
| 2 | 14.012475 | 32768/8 | Yes |
| 3 | 14.566805 | 32768/8 | Yes |

All **3/3** streams completed with HTTP 200, [DONE] and finish_reason=length;
HTTP, incomplete-stream and token-count failures were each **0/3**. The output hash
was dc5bcdac54c41a22c11643f735434b20c9079448f0b90a92b5234f70a0094140
for all three. The previous extra space was **not reproduced**, but three cold
controls neither prove determinism nor resolve the earlier equality failure.
The A/B/C diagnosis remains inconclusive and stopped at **2/21 requests**; these
three separately authorized controls do not fill its skipped pairs.

Fresh log suffixes contain exactly three snapshots and commits on every rank,
associated in request order with digest prefixes 8306fa87aa1e, 0df63e667fd5
and a73353fd864d. Each rank recorded **zero external hits or restores**. API logs
account for three completion POSTs and no other POSTs in the captured window.
The success counter moved **166→169**. Final checks passed with health 200,
idle 0/0, unchanged four-rank container/host/GPU process identities, runtime
configuration and source hashes, zero restarts and inactive flushers.

The private archive glm53-e09-cold-control-pending-20260913 contains a reviewable
**local-only candidate** derived from the installed corrected connector. It adds an
opt-in spark_cache_pending_wait_ms extra-config key, disabled by default; the
archived candidate setting of **2,000 ms** is an unqualified experimental deadline.
Known scheduler-issued publications receive fresh attempt tickets. A matching
replay waits with a fixed monotonic deadline and bounded state, then admits restore
only after **4/4 current worker generations** confirm that ticket and the existing
held-digest quorum. Error, unknown outcome, changed generation, lost quorum or
timeout leads to conservative recomputation. Ordinary cold/short misses bypass
waiting; failed requests do not restart their wait after late completion.

The candidate uses R10's existing zero-token connector steps to return worker
reports while a replay remains deferred, including its asynchronous batch queue.
It does not add an autonomous heartbeat when the engine has no requests.
Unrelated requests remain schedulable, ordinary decode carries no publication
probes, and waiting does not repeatedly hash the 32K prompt. The proposal leaves
the existing rank-synchronous restore-error recomputation path intact.

**59/59 CPU tests passed**: 22 new pending-publication cases and 37 existing store,
quorum, retirement, aggregation and serialization cases. They include a real
threaded page commit to disk, missing/delayed fourth reports, fixed timeout,
failure, generation/ticket changes, state limits and exact extracted R10
normal/asynchronous no-forward methods. Archived inputs reproduce the final pass.
These tests use lightweight engine interfaces; they do not qualify full distributed
GPU operation, output correctness or performance. The earlier one-space difference
remains an independent unresolved gate.

Candidate connector SHA-256 is
33b017b3b220b5914f68f08033aa641773be9437eaf583d51d1d16e978d5277c.
The archive retains the complete candidate, narrow patch, original and proposed
transfer configuration, tests, source pins, commands, payloads, raw receipts,
logs and root review. The original 447-entry benchmark archive and 131-entry
timing archive were reverified unchanged, as was frozen F0.
Any next live qualification needs a separately authorized coordinated transition
and targeted commit/report/admission traces, with the exact-output gate resolved
or explicitly adjudicated. **No candidate deployment, model load, restart, OFF
comparison, native benchmark, promotion, public API change or F0 restoration
occurred in this follow-up.** The current ON process remains loaded.

### Pending-publication live qualification window

The owner subsequently authorized a coordinated four-rank trial of the prepared
pending-publication candidate, with one weight load. Its initial read-only check
matched the saved corrected ON process, all source pins, health 200, idle 0/0 and
success counter 169. The candidate adds bounded state-transition traces to the
archived local patch, permitting direct association of publication, commit outcome,
report emission/receipt, admission and fallback without logging plaintext salts.
The trace revision passed **60/60 CPU cases**; seven offline checks of the unchanged
21-request diagnostic protocol also passed. Staging and dry-run comparison passed
on all four ranks, with only the connector mount and pending-wait config changing.

This window runs the two documented post-boot gates and, only after those pass,
a fresh **21-request A–B–C, B–C–A, C–A–B** series through the pinned Beast0 native
Rigmark client. Input remains 32,768 tokens with eight continuation tokens,
temperature zero and ignore_eos=true; every pair has a fresh nonce and salt.
HTTP, stream, count or cold/replay output failure stops subsequent probes without
replacements. Native performance benchmarking, OFF comparison and promotion are
outside this diagnostic qualification. The fixed F0 baseline is unchanged, and
automatic reference reloads remain suspended.

The candidate recipe and boot/rollback signatures are recorded in
[operations](docs/operations.md#e09-pending-publication-diagnostic-qualification).
Evidence is accumulated separately in glm53-e09-pending-live-20260913; the prior
247-entry cold-control/candidate archive was reverified without modification.

The single candidate up completed in **728.456 seconds**. Health reached 200 at
**20:22:45.919366 UTC on 2026-09-13**; both documented gates passed **2/2 within
3.223 seconds**, with normal Rome content and a valid get_weather call for Milan.
The new four-rank runtime matched every expected source, argument and mount.
The diagnostic series ran at **20:24:23–20:26:35 UTC** and completed **21/21 requests**.
HTTP, token-count, empty-output and incomplete-stream failures were each **0/21**;
cold/replay exact-continuation equality passed **9/9**. There were no replacements.

| Pair | Condition | Cold TTFT, s | Replay TTFT, s | Actual cold→replay, ms | Scheduler publication wait, ms | External hit |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | A | 13.485896 | 0.911770 | 1.223212 | 15.828679 | Yes |
| 2 | B | 12.771987 | 0.798859 | 1001.270542 | 6.417305 | Yes |
| 3 | C | 12.699091 | 0.790521 | 1001.843100 | 6.010395 | Yes |
| 4 | B | 12.681475 | 0.786115 | 1003.116199 | 5.298813 | Yes |
| 5 | C | 12.690565 | 0.802091 | 1002.882359 | 5.476044 | Yes |
| 6 | A | 12.661550 | 0.813271 | 2.352728 | 4.829759 | Yes |
| 7 | C | 12.691001 | 0.819547 | 1003.006471 | 6.649608 | Yes |
| 8 | A | 12.883449 | 0.973799 | 2.552279 | 5.199358 | Yes |
| 9 | B | 12.765475 | 0.809202 | 1003.155350 | 5.771227 | Yes |

External hits are **A 3/3, B 3/3, C 3/3**. Every physical rank logged nine verified
restores, nine snapshots and nine commits; all nine admissions followed current
ticket/generation confirmation from **4/4 ranks**. The three bridges started
250.544–250.856 ms after cold closure and finished before the replay deadline.
All six one-second replays were 1.271–3.155 ms late, within the predeclared 20 ms
tolerance; no sample required exclusion or replacement.

Targeted state traces distinguish both mechanisms. In pair 1, the scheduler
initially held ranks 1/2/3 and explicitly received a pending report from rank 0.
On Beast0's own clock, rank 0's committed callback followed wait entry by
13.720 ms; admission followed only after all four confirmations, at 15.829 ms.
In pair 2, even after the one-second delay, the scheduler held only ranks 0/1/2.
Rank 3's committed callback preceded its first ticket-specific report emission by
984.610 ms on that worker's clock; the solicited report reached the scheduler
5.117 ms after wait entry and admission followed at 6.417 ms. These within-node
intervals and causal message order demonstrate delayed visibility without
subtracting unsynchronized rank clocks or inferring scheduler state from commit
timestamps. They do not identify a network fault or every earlier ordinary report.

The outcome is **diagnostic qualification passed**, with a healthy candidate
left loaded. The final success counter was **23**: two boot gates plus 21 probes.
Health 200, idle 0/0, all four new process identities/source pins, zero restarts
and inactive flushers remained verified. A/B/C all hit, so rates alone do not
separate the causes; the state traces provide that evidence. The earlier one-space
divergence did not recur, but its original cause remains unresolved.
No live fault or timeout was injected: those fallback paths remain CPU-tested,
and the 2,000 ms deadline is not empirically tuned by 4.830–15.829 ms observed waits.
Native performance, the inherited C2 outcome and promotion remain unqualified.

### Pending-publication native series: stopped after run 1

The owner subsequently authorized three native Rigmark runs on the same loaded
candidate, with no new gates or weight load. Initial receipts matched the four
saved container, host/GPU process identities and source pins, health 200, idle 0/0
and counter 23. Rigmark 1.1.0 source/worktree/prompts and all F0 parameters matched;
three fresh independent run salts were prepared, with only the first used.
The unchanged native CLI completed run 1 at **21:34:58–21:48:57 UTC on 2026-09-13**.

The series stopped at **1/3 native runs, 54/162 requests**. Runs 2 and 3 were never
invoked; **108 requests were skipped**, with no replacement attempts. Native
receipt validation found zero schema/timing/hash errors and all 54 streams had
complete end markers. These results do not imply functional success: basic decode
completion passed **code 4/5, prose 5/5, structured 5/5**. Code row 2 degenerated
into a repeated boolean expression and reached 8192 tokens with finish_reason=length.
F0 had zero code caps in 15 requests. The output and its hash are retained privately.
The native invocation finished before the post-run assessment; this was not a
mid-run fail-fast interruption.

The rank-0 log also records **eight non-loopback chat admissions during the native
run**, in addition to its 54 loopback inference requests and 18 tokenizer calls.
The foreign admissions occurred at 21:42:00–21:44:39 UTC, overlapping prose/structured
and the approach to prefill. Their owner and payloads were not identified.
The counter rose **23→83**, versus the expected 77: native receipts account for
14 stop and 40 length completions, and counters show six additional stop completions.
Two extra HTTP admissions have no matching finish-counter increase; their outcome
is unknown. HTTP 200 headers do not prove complete streams. The capped code request
ran earlier, at 21:35:51–21:37:55 UTC, so the observed foreign traffic does not explain
that preceding failure. No cache publication transition was logged during code;
the cause of the repetitive output remains unresolved.

**Qualification verdict: discard for promotion in this series. Performance verdict:
unresolved.** The functional gate failed and the run was not exclusive. Preserve
the native values below without treating one contaminated run as a three-run median
or resolving the previous C2 tradeoff. Throughput is tokens/s; TTFT is seconds.

| Metric | Fixed F0 median | Candidate median of 3 native runs | Actual run 1, contaminated |
| --- | ---: | --- | ---: |
| Code decode | 50.401 | Not available | 52.535 |
| Code C1 aggregate end-to-end | 37.223 | Not available | 36.780 |
| Code C2 aggregate end-to-end | 57.009 | Not available | 51.729 |
| Code C4 aggregate end-to-end | 71.084 | Not available | 61.924 |
| Prose decode | 29.220 | Not available | 30.317 |
| Code TTFT | 0.410 | Not available | 0.413 |
| Prose TTFT | 0.397 | Not available | 0.383 |
| C1 per-stream TTFT | 0.404 | Not available | 0.357 |
| C2 per-stream TTFT | 0.616 | Not available | 0.469 |
| C4 per-stream TTFT | 0.935 | Not available | 0.513 |
| 8K cold prefill | 2112.218 | Not available | 2189.472 |
| 8K replay prefill | 4649.988 | Not available | 5351.013 |
| 32K cold prefill | 2201.970 | Not available | 2330.285 |
| 32K replay prefill | 21524.192 | Not available | 19937.234 |
| 64K cold prefill | 2201.282 | Not available | 2357.709 |
| 64K replay prefill | 35971.377 | Not available | 37596.262 |

All **9/9 native cold/replay outputs were identical**, with correct input counts,
eight continuation tokens and complete streams. All nine replays restored externally
on **4/4 ranks** after current ticket/generation confirmation. Each rank recorded
nine native snapshots, commits and restores; the raw interval additionally includes
four foreign chat publications and two foreign restores per rank. Digest and request
type/order association distinguishes these from the native observations. Cached
prefixes were 6912, 32256 and 64512 tokens, aligned to complete 2304-token blocks.
Scheduler waits were **5.219–575.334 ms**; no fallback or error/timeout injection occurred.
The private report presents all nine pairs. These cache results do not waive the
failed code gate or establish a performance gain. F0 remains frozen, including the
previously documented scheduling-description discrepancy; its macOS client also
differs from this Beast0 loopback Linux client.

Post-run receipts on **2026-09-14** retain all four process identities and source
pins, zero restarts, health 200, idle 0/0, inactive flushers and counter 83. The
rank-0 check correctly fails the expected count 77 after its identity/recipe checks
pass. Idle temperatures were 46/47/47/47 °C with no sampled thermal or power-brake
flags; no under-load thermal observation was captured for this series.
The candidate remains loaded under the owner's no-automatic-reference-reload
instruction; no serving configuration, public API or production default changed.

Before another performance series, diagnose the saved code-row-2 repetition and
establish an exclusive client window. Additional inference, live fault qualification
or a new candidate transition is a separate follow-up; do not add replacement runs.
Retain the correction's 4/4 requirement, bounded pending wait and recomputation on
error/timeout. Commands, native receipts, source/prompt pins, traffic accounting,
all-rank logs and the complete comparison are preserved in private archive
**glm53-e09-pending-native-20260913**. Use the unchanged pending-publication overlay
documented in operations for any eventual coordinated stop.

### Current service and evidence

**Reasoning diagnosis concluded, 2026-09-17 20:11 UTC — unresolved:** the
[qualification record](E09-QUALIFICATION-2026-09-17.md) supersedes the earlier
not-started state. Matched Low pilots complete 30 native requests; both thinking
values pass format 5/5 but fail Go correctness. True/High completes 15 requests,
with format 3/5, compilation 4/5, model tests 3/4 and independent/race suites 1/4;
the successful Go sample adds forbidden prose. The earlier interrupted Max
attempt had two length-capped completions and one interrupted client stream,
without an auditable raw receipt. False/Max and false/High remain unmeasured.
No profile passes; confirmation and the conditional DFlash transition do not
start. Owner conversation is permitted during this correctness diagnosis; its
timings are not performance evidence. Final 4/4 process/source checks pass,
health 200, restart 0, idle 0/0, counter 87. Rigmark is restored and fixed, 98 tests
and the infrastructure check pass. No reload, serving-configuration change,
reference restore or promotion occurred. Healthy target-only remains loaded.

**Code-output RCA completed, 2026-09-17:** the detailed
[analysis and remediation plan](E09-ROOT-CAUSE-ANALYSIS-2026-09-17.md) identify
forced thinking-off with default Max effort as the first causal hypothesis.
The configuration mismatch is verified in 40 CPU render/parser combinations;
whether it causes the observed revisions still needs an A/B comparison. The
same adapter and flag existed in F0, preventing attribution to a new E09 change.
Offline replay of saved target-only Go answers passes model-supplied tests 3/5;
samples 4/5 fail, including the fifth answer's revised test. Native basic gates
do not measure these requirements, and the separate audit extractor has
reproduced filename-fence/prose errors. Fix measurement in Rigmark, then test
thinking parameters on the loaded process before considering engine changes.
No new inference, restart or serving mutation occurred. At 16:10 UTC the same
four target-only containers remain healthy, restart 0, idle, counter 17. The
plan is prepared; no next experiment is running or qualified.

**Target-only diagnostic completed, 2026-09-17 13:50 UTC:** after the additional
attempt reproduced the code anomaly, the owner authorized comparison without
DFlash. Four-rank dry-runs remove only `--speculative-config`; source, GID and
jumbo checks pass. The original stack was stopped in one coordinated transition;
one new target-only load passed Rome/tool gates 2/2 within 3.832 seconds of
health. Live logs confirm speculative_config=None, engine_k=0, adaptive policy
disabled and async=True. Native decode-only completed 15/15 requests (five code
plus ten prose/structured controls), with all basic gates passing, no foreign
traffic and counter 17. Code format still passes only 2/5: samples 2/4 append
prose, sample 5 repeats its test file after self-correction. None reaches the
length cap. The defect occurs without DFlash; disabling it is insufficient,
and the root cause remains unresolved. All five reasoning fields are empty;
no external cache activity appears on any rank. Final identity/source checks
pass on healthy idle 4/4 processes with zero restarts and no active native job.
This is a bounded correctness diagnosis, with no three-run performance median
or semantic Go execution audit. Use the target-only overlay in operations for
eventual down; earlier process identities below are historical. No automatic
F0 reload or promotion.

**New exclusive series, 2026-09-17 11:31 UTC:** the owner authorized three new
native runs on the same loaded processes after the earlier exclusive series
stopped. Fresh 4/4 identity/source/idle checks pass, with success counter 56.
Run 1 started at 11:31:50 UTC in `native-exclusive-20260917-1130`; labels, salts
and the traffic boundary are new, while the engine and native protocol are
unchanged. Expected counters are 56 → 110 → 164 → 218. No additional boot,
functional-gate requests or warmup was performed. Run 1 completed at 11:42:59
UTC with native integrity PASS, 54/54 complete streams, basic code gates 5/5,
equal replays 9/9 and verified restores on 4/4 ranks. Later manual review found
code sample 5 repeated its implementation/test blocks, emitting seven blocks
plus self-revision prose instead of the two blocks and no prose requested.
The native gate does not check those requirements. The format review passes
4/5; this is distinct from semantic code execution or the earlier length cap.
Run 2 was already underway when reviewed and its owned client was interrupted
at 11:47:53 UTC after six completed requests and one interrupted stream. No
run-2 JSON or partial bodies were persisted by the native client. Run 3 did not
start. Final counter 116 and idle healthy 4/4 identities passed; traffic remained
exclusive. No three-run median is available.

**Owner-requested additional attempt, 11:53 UTC:** the owner asked to retry
before declaring the experiment failed as a whole. One new complete 54-request
native run started at 11:53:25 UTC in `native-confirmation-20260917-1153`, on the
same processes and protocol, with a fresh label/salt and counter 116 → 170.
The confirmation completed at 12:06:03 UTC with 54/54 complete streams, zero
native receipt-validation errors, no foreign traffic and counter 170. All five
code outputs were reviewed: the native basic gate passes 4/5, and only 2/5 meet
the requested two blocks without prose. Sample 4 repeats full implementations
and tests until the 8192-token cap; samples 1/2 add self-revision prose and extra
blocks. This reproduces the output anomaly in the additional attempt. It does
not establish its cause or prove a regression attributable to any specific
engine, cache, transport or numerical component. No semantic Go execution audit
was performed. The nine cache replays have equal continuations and verified
restores on every rank, with bounded waits 5.012–67.424 ms.

The fresh runtime check at 12:06:49 UTC passes on unchanged healthy 4/4
processes, zero restarts, all source pins, idle 0/0 and inactive flushers. No
benchmark remains active. The replacement and confirmation produce two complete
native receipts (108 requests), plus six completed requests and one interrupted
stream in the partial run. Across complete receipts only: basic code gates 9/10,
two-block/no-prose review 6/10, code caps 1/10, equal replay pairs 18/18. Partial
output bodies were not persisted and are excluded from those review totals.

Verdict **unresolved**; candidate not qualified for promotion. No three-run
median is claimed. Single-run code/C1 values improved over fixed F0, while C2
was 48.523 and 53.471 tok/s versus F0 57.009; correctness prevents qualification
regardless of speed. The unchanged candidate remains loaded under the owner's
no-automatic-reference-reload instruction. F0 and all earlier evidence remain
unchanged and separately reported.

**Recovery preparation on mini, 2026-09-17:** inspection reproduced the stale
explicit SIRCL selections in the inherited runtime: rank 1 GID0=4 and rank 3
GID1=4, while all eight selected ports currently have the matching IPv4 RoCEv2
entry at 3. The complete bundle/runtime hashes agree across four nodes; selected
links are ACTIVE/LinkUp, 200 Gb/s and MTU 9000. No host/network change was made.
The new `gid-recovery-20260917` runtime corrects those two entries and validates
the explicit selection before vLLM imports. Ten CPU tests and four exact-image
weight-free capability probes pass; the latter also reproduce the previous
two-rank rejection with the old settings. Four dry-runs change only the runtime
mount. The initial probe's missing `ip` utility is preserved as a prerequisite
failure and resolved using read-only interface ioctls. A supervisor absence-message
case mismatch also stopped before any controller call and was corrected with the
actual Docker response retained as a regression case. The real coordinated boot
then completed in 691.385 seconds with one weight load; `/health` reached 200 at
06:22:33.048 UTC and both functional gates passed in 3.013 seconds. Runtime identity,
41 source files per rank, effective SIRCL settings, mapped libraries, idle 0/0,
inactive flushers and counter 2 passed on 4/4 nodes. The GID boot failure is resolved.
The first native run completed at 06:34:36 UTC: 54/54 complete streams, zero native
validation errors, basic code/prose/structured gates 5/5 each, zero code length
caps, and 9/9 equal cold/replay continuations. All nine cache hits have current
generation confirmations from 4/4 ranks and verified restores on every rank;
bounded scheduler waits are 4.911–52.833 ms. The measured interval has exactly
the expected loopback POSTs and no foreign traffic.

After the local session interruption, the next check found one non-loopback chat
POST at 09:49:57 UTC, over three hours after the native finish. It returned 200
and caused model activity and a 52,992-token cache restore. The full-series
traffic gate failed before run 2; preserve that failure separately from the
clean bounded run interval. The later success counter remains 56, which does
not establish the foreign request's outcome or exclusivity. Runs 2/3 were not
launched: **1/3 runs and 54/162 native requests**, no three-run median,
**unresolved**, no promotion. The prior code degeneration was not reproduced in
five new outputs; its cause remains unresolved. Native basic output gates do
not establish semantic code correctness.

Single-run observations only: code 53.512 tok/s, C1 40.464, C2 56.728 and C4
70.752, against fixed F0 medians 50.401, 37.223, 57.009 and 71.084. These do not
establish repeatable gains or a primary-metric tradeoff. F0 remains frozen;
the private summary retains all 16 fixed-baseline metrics and explicitly absent
three-run variant medians. Historic F0 used macOS, while this client uses Linux
loopback; the recorded F0 async-description caveat still applies.

The final check confirms unchanged healthy 4/4 processes, zero restarts,
all source pins, idle 0/0 and inactive flushers. No native job remains active.
The candidate stays loaded under the owner's no-automatic-reference-reload
instruction. Full recipe pins, retained receipts and stop guidance are in
[operations](docs/operations.md#e09-sircl-gid-recovery-on-2026-09-17).

**Earlier observation, 2026-09-17 05:36:45 UTC:** the owner approved the prepared
restart and three new native runs. Its single boot attempt failed before health:
SIRCL rejected unavailable RDMA GID `rocep1s0f0:4` on rank 1 and
`rocep1s0f1:4` on rank 3. No gate or benchmark request was sent: **0/3 runs,
0/162 native requests**. After saving four-rank logs, the existing supervisor
performed coordinated down successfully. All four nodes have no running
containers or GPU processes and inactive flushers. No reference was reloaded.

The owner then requested investigation through resolution, followed by a handover
to continue from another computer. The root cause below the SIRCL capability
rejection remains open: actual sysfs GIDs and the loaded SIRCL selection must be
checked before changing indexes or loading weights. NCCL automatic selection
and successful jumbo pings do not validate the separate SIRCL selection.
The private `HANDOVER-E09-2026-09-17.md` delivered to the owner's mini records
the exact authorized scope, pins, portable archive
**glm53-e09-pending-restart-20260917**, completed
cleanup and remaining work. The previous code-output failure is still unresolved.

**Earlier resume check, 2026-09-17:** the owner asked to resume after an Internet outage.
The four SSH targets are reachable, but the candidate process saved on September 14
no longer exists. Ranks 0/1 have no Docker containers. Ranks 2/3 retain different
containers using the v11 DFlash2 image, started on September 16 around 19:03 UTC and
exited with code 1 around 19:13 UTC. All four GPUs have no compute processes,
flushers are inactive, and the rank-0 API refuses connections. The read-only check
stopped before inference: **0 additional runs, 0 requests, 0 lifecycle actions**.
Neither the intervening service changes nor the previous code failure are explained
by the owner's Internet-outage report alone.

The intended E09 R10 image remains present on all four nodes; all ten source pins
per rank match. Four read-only launcher dry-runs exactly reproduce the qualified
candidate arguments, the pinned controller is intact, and the native Rigmark client
still matches F0 source/worktree/prompts. No remote run-2/run-3 artifacts exist.
The available recipe is a prepared candidate, not a currently serving overlay.
Evidence and the restart proposal are in **glm53-e09-pending-resume-20260917**.

Resuming the original loaded-process series is impossible. The proposed next window
is one coordinated E09 boot with one weight load, the fabric prerequisite and two
timely functional gates, then three new native Rigmark runs (162 requests) on the
new process. Keep the contaminated September 13 run and its failed code gate as
separate evidence; do not silently replace it or merge different-process results
into the original three-run median. That window was subsequently approved and
failed before inference as recorded above. No reference reload, promotion,
host/network change or automatic retry was performed.

The following paragraphs preserve the earlier lifecycle history.

The previous attempt-5 E07a restoration completed at 12:10:13 UTC on 2026-09-13. Coherent and tool-call gates
passed 2/2 within 8.801 seconds of health. Final receipts show one exact stack per rank,
restart counts 0, idle 0/0, two completed gate requests and 49/49/48/49 °C, with sampled
thermal and power-brake flags inactive. These are post-restore observations, not a
continuous R10 thermal trace. During the subsequent named diagnostic boot, a sample
showed 48/49/48/51 °C and inactive thermal and power-brake flags.

After the failed named diagnostic request, E07a restoration began. The owner then
explicitly requested continuous root-only diagnosis, accepted extended cluster
downtime and rejected repeated reference unloading/reloading. The exact owned
watcher and E07a up controller were cancelled; its TERM cleanup stopped all four
ranks. The cancelled restore took 549.664 seconds, sent no gate requests and is not
a successful restoration. All-four stopped state was verified before the isolated
operator tests. The owned test container was then removed, the fabric check passed,
and the clean mapper-corrected R10 process completed the OFF diagnosis above.
Following the reproduced second defect, that process was stopped once for the
prepared corrected SparkCache ON candidate. Automatic reference restoration is
suspended for this owner-authorized continuous diagnostic window.

The final attempt-5 report preserves the superseded drafts, explicit 0/3 native runs
and 0/162 requests, fixed F0 medians with unmeasured variant values, commands and
restoration evidence. Private receipts and candidate payloads remain outside the
checkout. The current investigation has not changed production defaults or promoted
R10, SIRCL or SparkCache. Lifecycle boundaries remain those in
[operations](docs/operations.md#experimental-e09-jj-r10-window).

## Protocollo pianificato per E01

Il protocollo seguente resta come record riproducibile, ma E01 è stata fermata e scartata
dal proprietario prima di completare i tre run; non deve essere ripresa automaticamente.

### Riferimento e unica variabile

E01 usa i valori F0 congelati in [`docs/baseline-f0.json`](docs/baseline-f0.json) senza
rieseguire la baseline e senza modificare quel file. L'evidenza operativa archiviata di
F0 riporta `async_scheduling=true` e la classe personalizzata deriva da
`AsyncScheduler`, mentre il flag CLI opzionale `--async-scheduling` era assente; la
descrizione `async scheduling disabled` nel JSON congelato è inesatta. E01 deve
riprodurre il lancio F0 effettivo senza aggiungere quel flag e deve registrare la
discrepanza nel proprio archivio.

La sola variabile è `VLLM_ADAPTIVE_K_MODE=batch-uniform` al posto di `per-request`.
Restano invariati `k_lo=3`, `k_hi=5`, checkpoint FP8, drafter DFlash2, immagine, motore,
argomenti, ambiente, scheduler, graph, MoE, NCCL, topologia e parametri Rigmark.

### Sequenza autorizzabile

1. Usare una finestra operativa separatamente autorizzata. Confermare identità F0,
   assenza di carichi estranei e archivio di destinazione fuori dal checkout. Non
   eseguire alcun run F0.
2. Applicare la sola variante di modalità a tutti e quattro i nodi con una transizione
   coordinata. Caricare i pesi una sola volta e mantenere lo stesso processo del modello
   per tutti e tre i run; non riavviare o ricaricare tra i run.
3. Dopo `/health` 200, completare entro due minuti il controllo di risposta coerente
   (Roma) e il controllo tool-call descritti in
   [`docs/operations.md`](docs/operations.md#post-boot-functional-gates). Se falliscono,
   fermare l'intero stack e chiudere l'esperimento come fallito.
4. Eseguire tre invocazioni native consecutive `./rigmark run`, con la stessa versione,
   fingerprint, prompt, seed, temperatura, `top_p`, condizioni idle e politica cache di
   F0. Non eseguire run Rigmark di warmup separati.
5. Ciascun run completo contiene esattamente 54 richieste, per 162 richieste totali:
   - 15 decode: cinque richieste per `code`, `prose` e `structured`, limite 8192 token;
   - 18 prefill: per 8K, 32K e 64K, tre coppie cold/replay per profondità;
   - 21 concorrenza: tre round del workload codice a C1, C2 e C4, 256 token per stream
     (`1 + 2 + 4` richieste per round).
6. Validare le tre ricevute native, i 54 incrementi dei contatori per ogni run, gli output
   completi, gli errori e l'identità invariata del processo. Conservare comandi,
   configurazione, ricevute,
   fingerprint, log, gate e hash nell'archivio esterno al checkout.
7. Confrontare la mediana dei tre valori prestazionali per-run di E01 con la mediana F0
   congelata. Riportare colonne `F0 mediana fissa` ed `E01 mediana di tre run`, i conteggi
   di 3 run e 162 richieste; correttezza ed errori restano conteggi espliciti con
   denominatore. Non aggiungere percentuali o inferenza statistica non prodotte dalla
   misura.
8. Ripristinare F0 con una transizione coordinata completa, attendere `/health` 200 e
   ripetere i due gate entro due minuti. Chiudere la connessione e l'archivio, presentare
   il report e fermarsi: nessun esperimento successivo parte automaticamente.

E01 non promuove automaticamente IaC, non modifica la ricetta di produzione e non
autorizza commit o push. Un eventuale `promote` richiede una decisione successiva del
proprietario e l'intero percorso di promozione definito in `AGENTS.md`.
