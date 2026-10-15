# Progetto Banca Virtuosa – XAI Computer Vision
## Riepilogo del brainstorming (fase 1)

**Data:** 9 ottobre 2026
**Natura del documento:** riepilogo strutturato della discussione, **non una trascrizione letterale**. Le decisioni definitive e la specifica completa sono nel notebook `profai_11_XAI_prj_banca_virtuosa_design_v1.ipynb`. I valori numerici indicati come "attesi" sono stime fatte durante la discussione, da misurare in fase zero.

---

## 1. Processo seguito

Il brainstorming ha seguito la skill `brainstorming` di superpowers (obra/superpowers):

1. capire il contesto e valutare lo scope;
2. fare **una domanda alla volta**, preferibilmente a scelta multipla;
3. proporre 2-3 approcci con trade-off e una raccomandazione;
4. presentare il design **a sezioni**, con approvazione dopo ciascuna;
5. verifica finale (segnaposto, contraddizioni, ambiguità, scope);
6. documento finale. Nel tuo flusso è il notebook Colab di design della fase 2, da dare in input a una nuova chat (la skill prevede poi la skill `writing-plans`).

**Scope:** un solo progetto, scomposto in 6 unità isolate: dati/preprocessing, modello (adattamento, training, checkpoint), baseline e modello spiegabile (LR), tecniche XAI, valutazione e concordanza, report.

---

## 2. Valutazione iniziale dei 12 punti di analisi preliminare

| # | Punto | Valutazione |
|---|---|---|
| 1 | Demo: no augmentation, tuning minimo | D'accordo. Va dichiarato nel report come scelta consapevole, con seed fisso. |
| 2 | MNIST + DenseNet da torchvision | D'accordo. Fissare la variante: DenseNet121, pesi `IMAGENET1K_V1`. |
| 3 | LR come modello spiegabile | Criterio *interpretable-by-design* solido. **Da chiarire:** una LR addestrata sulle etichette vere spiega se stessa, non la DenseNet. Un surrogato allenato sulle predizioni della DenseNet sarebbe un'approssimazione post-hoc. Note tecniche: coefficienti multinomiali identificabili solo a meno di una costante tra classi (usare `w_c − media`); mappa locale esatta `w_c ⊙ x`. |
| 4 | Niente feature morfologiche | D'accordo. |
| 5 | LR come baseline | D'accordo; stessi split e stessa normalizzazione di partenza. |
| 6 | Metriche | D'accordo. Precision/recall/F1 macro e per classe, confusion matrix, tempi di training, inferenza **e spiegazione**. Serve uno split train/val/test (54k/6k/10k). |
| 7 | Grad-CAM di debug, prima e ultima epoca | D'accordo. Stesse immagini fisse di validation. "Ultima" = ultima epoca eseguita (early stopping). |
| 8 | Classe di config | Idea giusta. **Correzione:** in un `dataclass` gli attributi senza annotazione (`NUM_EPOCHS = 5`) non sono campi. Il commento su GPT2 è un residuo. I moduli ricevono la config come argomento. |
| 9 | Spearman e IoU sotto flag | Le specifiche chiedono "quanto le tecniche concordano": la misura quantitativa ha senso. Decisione finale: `COMPUTE_AGREEMENT=True`. Cautele: pari merito e sfondo, soglia parametrizzata, concordanza ≠ fedeltà. |
| 10 | Resize a 3×128×128 | D'accordo sulla dimensione, **motivazione corretta:** il dettaglio di Grad-CAM dipende dallo stride 32 della DenseNet (128 px → 4×4; 224 → 7×7; 96 → 3×3), non dall'ingrandimento. Attenzione allo spazio normalizzato per baseline di IG, riempimento di Occlusion, background di SHAP. |
| 11 | Early stopping + checkpoint | D'accordo. Checkpoint con config ed epoca; fallback al training se manca. ~30 MB, può stare nel repo. |
| 12 | Moduli `.py` su GitHub + notebook sottile | D'accordo, con cautele: tag/commit fissato nel clone, `requirements.txt` senza torch, repo accessibile, notebook narrativo per i non tecnici. |

Rischio tecnico aggiuntivo segnalato: i ReLU `inplace=True` della DenseNet di torchvision rompono hook e backward di Captum/SHAP.

---

## 3. Domande e decisioni

### Q1 – Ruolo della regressione logistica
- **A.** Solo baseline e sistema alternativo spiegabile, allenato sulle etichette vere.
- **B.** Solo surrogato globale della DenseNet (allenato sulle sue predizioni).
- **C.** Entrambi, come due istanze distinte.
- **Variante D (proposta su tua richiesta):** LR-base sempre attiva (etichette vere), con mappe globali e locali, confronto a **quattro quadranti** con la DenseNet e flag `USE_SURROGATE` (spento) per la fedeltà.

**Scelta: D.** La LR non spiega la DenseNet, e il report lo dichiara.

### Q2 – Strategia di fine-tuning
- **A.** Solo la testa (backbone congelato).
- **B.** Rete intera con learning rate basso (inizialmente raccomandata).
- **C.** Due fasi (gradual unfreezing).

Hai obiettato che il progetto non mira a migliorare le prestazioni ma a capire dove il modello sbaglia, quindi la B non era giustificata. Stime discusse: A ≈ 96-98,5 % (da misurare), LR ≈ 92-93 %; con la A ci sono più errori da analizzare e il training è più veloce.

**Scelta: A come default (`FREEZE_BACKBONE=True`).** Se l'accuracy di validation è sotto `MIN_VAL_ACCURACY` compare un **avviso** e l'esecuzione prosegue; la decisione di sbloccare la rete intera resta all'operatore (`FREEZE_BACKBONE=False`). Il fallback automatico su `denseblock4` proposto inizialmente è stato ritirato.

### Q3 – Librerie XAI
- **A.** Captum per tutte e cinque.
- **B.** Mix: Captum per Grad-CAM/IG/Occlusion, pacchetti originali `lime` e `shap` per LIME e SHAP.
- **C.** Solo pacchetti dedicati.

Hai chiesto di non decidere prima di aver fatto delle prove.

**Scelta: backend intercambiabili** (`LIME_BACKEND`, `SHAP_BACKEND`), default provvisorio = pacchetti originali, **decisione dopo la fase zero**. Nessun fallback automatico silenzioso.

### Q4 – Selezione degli esempi da spiegare
- **A.** Un solo campione piccolo.
- **B.** Due livelli: galleria qualitativa (~24 immagini scelte) + campione quantitativo stratificato di `N` immagini.
- **C.** Test set intero per le tecniche veloci.

**Scelta: B.** Per gli errori si spiegano sia la classe predetta sia la classe vera.

### Q5 – Comportamento di "Esegui tutto" rispetto al training
- **A.** `load` di default. **B.** `train` di default. **C.** `auto`.

Chiarimento: `train` allena da zero e salva il checkpoint; `load` carica il checkpoint senza allenare; `auto` è la regola che sceglie tra le due.

**Scelta: C.** `auto` carica solo se il checkpoint esiste ed è compatibile con la config; altrimenti allena. Mai uso silenzioso di un checkpoint incompatibile.

### Q6 – Layer di Grad-CAM
- **A.** `features.norm5` (4×4). **B.** `features.denseblock3` (8×8). **C.** Parametrizzato, default A, più una figura di confronto.

**Scelta: C.**

### Q7 – Griglia comune e segno delle mappe
- **A.** 28×28, solo evidenza positiva. **B.** 128×128, valore assoluto. **C.** 28×28 con `SIGN_MODE` parametrizzato.

**Scelta: C.** Successivamente hai sollevato il dubbio sul valore assoluto (prassi abituale nel confronto tra mappe). Il default è stato cambiato in **`SIGN_MODE=absolute`** (alternativa `positive`). Le figure restano con segno; il report dichiara che Spearman e IoU misurano l'accordo sull'**importanza**, non sul verso dell'influenza.

### Q8 – Su quali pixel calcolare Spearman e IoU
- **A.** Tutti i 784 pixel. **B.** Solo la cifra (con dilatazione). **C.** B come default, con `MASK_MODE` e tabella dell'attribuzione fuori dalla cifra.

**Scelta: C.** Chiarimento sulla domanda "perché attribuzione sullo sfondo se è sempre nero?": può derivare da (1) artefatti della tecnica (Grad-CAM 4×4 ingrandito, finestre di Occlusion, celle di LIME), (2) il riferimento usato (baseline di IG, baseline casuali di GradientShap, coerenza con la normalizzazione), (3) evidenza legittima (l'assenza di inchiostro può essere informativa). Nel report non si scrive "la rete guarda lo sfondo".

### Q9 – Misura della fedeltà (curva di deletion)
- **A.** Non includerla. **B.** Dietro flag, spento. **C.** Accesa di default.

**Scelta: B** (`COMPUTE_FAITHFULNESS=False`). Motivo: concordanza ≠ correttezza.

### Q10 – Output del report
- **A.** Notebook narrativo unico. **B.** Notebook sottile + report generato. **C.** Notebook narrativo + `outputs/` con figure (PNG) e tabelle (CSV) dai nomi stabili.

**Scelta: C.**

### Q11 – Test
- **A.** Test di fumo per modulo. **B.** Test estesi. **C.** Nessun test separato.

**Scelta: A.** Cella opzionale `RUN_TESTS`; lo stesso test sul contratto delle mappe gira su entrambi i backend.

---

## 4. Chiarimenti tecnici emersi lungo il percorso

- **Cosa significa "backend":** è la **libreria che calcola la mappa**, non la mappa. Vale per LIME (`lime` o Captum) e SHAP (`shap` o `GradientShap` di Captum). Grad-CAM, IG e Occlusion usano sempre Captum.
- **BatchNorm in `eval` con backbone congelato:** `requires_grad=False` ferma i pesi, ma non le statistiche (`running_mean`, `running_var`) che in `train()` si aggiornano a ogni batch. Si applica `eval()` **solo ai layer BatchNorm del backbone**, e va riapplicato a ogni inizio di epoca (`enter_train_mode`), perché `model.train()` rimette tutto in train. Per la testa (`Linear`, nessun dropout) `train()` ed `eval()` coincidono.
- **Numero di epoche:** 5 di default; si rivede dopo la prima curva (8-10 solo se la loss di validation scende ancora in modo netto). Con 5 epoche e patience 2 l'early stopping scatterà di rado.
- **Perché la LR non è standardizzata:** i pixel sono già in [0,1]; i pixel quasi costanti amplificherebbero il rumore; soprattutto la mappa locale `w_c ⊙ x` smetterebbe di essere zero sullo sfondo. Da verificare in fase zero (convergenza, ~92 %).
- **`lbfgs`:** algoritmo di ottimizzazione della LR in scikit-learn (parametro `solver`). `max_iter` troppo basso produce un `ConvergenceWarning`.
- **Parametri della LR in config:** `LR_PENALTY` (`l2`), `LR_SOLVER` (`lbfgs`), `LR_C` (1,0), `LR_MAX_ITER` (1000). `lbfgs` accetta solo `l2`; L1 ed elastic net richiedono `saga`. Validazione della combinazione. `LR_C` si regola solo sulla validation.
- **Pari merito (Spearman e IoU):** pixel con valore identico (celle di LIME, finestre di Occlusion, anello a zero della LR locale). Spearman usa ranghi medi; per l'IoU si usa un ordinamento stabile (a parità, ordine riga per riga) e il report segnala quando la soglia cade dentro un gruppo di pari merito.
- **Parametri rivisti:** `GRADCAM_LAYER=features.norm5`, `LIME/SHAP_BACKEND` provvisori, `USE_SURROGATE=False`, `SIGN_MODE=absolute`, `MASK_MODE=digit`, `COMPUTE_AGREEMENT=True`, `COMPUTE_FAITHFULNESS=False`, `RUN_TESTS=False`.

---

## 5. Design presentato (4 sezioni, tutte approvate)

1. **Architettura e struttura del repo:** moduli `config`, `data`, `model`, `training`, `baseline`, `metrics`, `sampling`, `xai/` (contratto + tecniche + backend), `comparison`, `reporting`, `tests`, `phase0_checks`, `main.ipynb`; flusso dati in 8 passi; lingua: codice in inglese, narrazione in italiano.
2. **Dati, modello e training:** split 54k/6k/10k; vista DenseNet (3×128×128, normalizzazione ImageNet) e vista LR (28×28, [0,1]); patch dei ReLU in-place; BatchNorm in `eval`; early stopping; checkpoint con snapshot dei parametri; Grad-CAM di debug.
3. **Tecniche XAI, confronto e fase zero:** contratto comune delle mappe; scelte per le cinque tecniche; pipeline del confronto (griglia 28×28, segno, maschera, Spearman, IoU, matrici 6×6, riferimento casuale, tabella fuori-cifra, deletion opzionale); 10 verifiche di fase zero.
4. **Report, rischi ed esclusioni:** struttura in 10 sezioni del report; rischi e limiti da dichiarare; esclusioni di proposito (YAGNI).

---

## 6. Verifica finale: cinque correzioni incorporate

1. **Debug Grad-CAM in modalità `load`:** mappe di debug e curva di training salvate accanto al checkpoint e ricaricate; se mancano, la sezione dichiara "non disponibile".
2. **"Ultima epoca" vs pesi finali:** il debug mostra l'ultima epoca eseguita, l'analisi usa la migliore epoca; il report riporta entrambi i numeri.
3. **Classe usata nel confronto:** tutte le mappe usano la classe predetta dalla DenseNet (la LR usa `w_a ⊙ x`); nei quadranti si mostra anche la classe predetta dalla LR.
4. **Learning rate:** due costanti, `LEARNING_RATE_HEAD` (1e-3) e `LEARNING_RATE_FULL` (5e-5), scelte in base a `FREEZE_BACKBONE`.
5. **Pari merito:** ranghi medi per Spearman, ordinamento stabile per l'IoU.

Conferme: lingua (codice in inglese, narrazione in italiano); il budget di tempo di "Esegui tutto" non si fissa ora, ma dopo le misure della fase zero.

---

## 7. Prossimi passi

1. Aprire una **nuova chat**, allegare il notebook `profai_11_XAI_prj_banca_virtuosa_design_v1.ipynb` e usare il prompt della sezione 17 del notebook (skill `writing-plans`).
2. Partire dalla **tappa 0 (fase zero)** e fissare i valori ancora aperti: backend di LIME e SHAP, `N_QUANT`, `NUM_EPOCHS`, `MIN_VAL_ACCURACY`, patience, batch size, learning rate, `LR_C`, `LR_MAX_ITER`, soglia e dilatazione della maschera, parametri di LIME, SHAP e Occlusion.
3. Proseguire con le tappe 1-4 (modelli e metriche, tecniche XAI, confronto, report e notebook narrativo).
