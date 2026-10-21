# Errata al design v1 – XAI Banca Virtuosa

Il notebook di design originale non viene modificato. Questo file raccoglie le correzioni e le
precisazioni approvate dall'operatore e quelle ancora da confermare. Stato: **A** = approvata,
**P** = proposta (da confermare), **V** = da verificare in fase zero.

## Modifica richiesta

| ID | Modifica | Stato |
|---|---|---|
| D03-bis | Rimosso il modello surrogato. Spariscono `USE_SURROGATE` (config e tabella parametri §5), il "surrogato opzionale" in `baseline.py` (§4), il paragrafo "Surrogato" e la metrica di fedeltà come % di accordo (§8), il riferimento in §13 punto 8. Aggiunto alle esclusioni di §15. Restano: l'accordo tra le predizioni di DenseNet e LR nei quadranti (statistica descrittiva) e la fedeltà come deletion (D15). | A |

## Correzioni e precisazioni

| ID | Punto | Decisione | Stato |
|---|---|---|---|
| E01 | `MIN_VAL_ACCURACY` valeva 0,94 nella config e 0,95 nella tabella "da fissare" | **0,95** ovunque (decisione dell'operatore, dopo il run da 5 epoche con BatchNorm in `eval`: 0,9732 sul test). È un avviso, non uno stop, e si misura sull'accuracy di validation dell'**ultima** epoca (non c'è più selezione della migliore, vedi E20). | A |
| E02 | Coefficienti LR relativi (globale) e assoluti (locale) | Il design resta: globale `w_c − media`, locale `w_a ⊙ x` esatta sul logit. Con L2 la somma dei coefficienti sulle classi è ≈ 0, quindi le due versioni coincidono. Aggiunto un test diagnostico. | A |
| E03 | "Riduzione a 28×28" ambigua | Tutte le mappe a 128×128 (bilineare se più piccole, es. Grad-CAM 4×4), poi media su area a 28×28. | P · V (check 12) |
| E04 | `config.py` previsto in tappa 1 ma necessario in tappa 0 | Anticipato in tappa 0, rifinito in tappa 1. | A |
| E05 | Snapshot del checkpoint (D08) incompleto | Aggiungere `VAL_SIZE`, `BATCH_SIZE`, `NUM_EPOCHS`, `OPTIMIZER`, learning rate effettivo e `FLAG_BATCHNORM_EVAL` (cambia i pesi). `FLAG_RELU_INPLACE` no: non cambia i pesi. Un checkpoint con altri valori non è "compatibile". Il checkpoint di fase zero già salva questi campi. | P (tappa 1) |
| E06 | Test "senza GPU, dati minuscoli" contro pesi ImageNet | I test usano `weights=None` o un modello giocattolo. I pesi veri servono solo in fase zero e nel run reale. | A |
| E07 | Campione quantitativo "stratificato per classe" | Per classe **predetta**, N/10 per classe. | P |
| E08 | Composizione della galleria da 24 | 10 corrette (una per classe, a caso con seed, non scelte per bellezza) + 6 errori più confidenti + 8 dai quadranti (2 per quadrante, senza duplicati). | P |
| E09 | Maschera della tabella "fuori dalla cifra" | Maschera `digit` con dilatazione, indipendente da `MASK_MODE`. | P |
| E10 | Classe target del Grad-CAM di debug (D09) | Classe predetta, con l'etichetta vera nel titolo. | P |
| E11 | Arrotondamento del top-k (D22) | `k = ceil(TOPK_FRACTION · pixel nella maschera)`. | P |
| E12 | Aggregazione sui canali | IG e SHAP: somma. Occlusion: finestre su tutti e 3 i canali, poi media. **Verificato in `--quick` (check 11): le finestre sovrapposte sono mediate da Captum, le mappe dei 3 canali sono identiche, 226 forward = 225 finestre + l'originale.** | A |
| E13 | Modalità `load` | Rete costruita con `weights=None` e caricamento dello state dict, senza scaricare ImageNet. | P |
| E14 | Testo narrativo e risultati dinamici | Si salva lo SHA-256 del checkpoint. Se non coincide con quello a cui si riferiscono le interpretazioni scritte a mano, il notebook mostra un banner visibile. | P (tappa 4) |
| E15 | `pip install` su Colab può forzare un riavvio del runtime | Il check 1 lo rileva; la cella 1 del notebook asserirà le versioni dopo l'installazione. Nessuna correzione automatica. | V (check 1) |
| E16 | Validazione della config incompleta | Estesa a nomi dei backend, intervalli, coerenza griglia/finestra, `N_QUANT` multiplo di `NUM_CLASSES`. Valore non valido → errore, nessun fallback. Implementata in `config.py`. | A |
| E17 | Dilatazione della maschera: elemento strutturante non specificato | Quadrato 3×3 (8-connesso), `MASK_DILATION` iterazioni. | P · V (check 8) |
| E18 | `SHAP_NUM_SAMPLES` mancava: i campioni di gradiente erano un numero magico | Aggiunto alla config (default 200, come shap). | P |
| E19 | `IMAGE_SIZE` indicava due cose (MNIST 28×28 e input DenseNet 128×128) | Due parametri: `IMAGE_SIZE = 28` (MNIST, input della LR) e `DENSENET_SIZE = 128`. Le validazioni su griglia LIME, finestra Occlusion e stride 32 usano `DENSENET_SIZE`. Preso dal config dello script esplorativo dell'operatore. | A |
| E20 | Early stopping (D07) | **Eliminato** (con 5 epoche complica il codice senza vantaggio). Via `EARLY_STOPPING_PATIENCE`, via il relativo test e la sua menzione in §7. Numero di epoche fisso; si salvano i pesi dell'**ultima** epoca. | A |
| E21 | Due nuovi flag in config | `FLAG_BATCHNORM_EVAL = True`: BatchNorm del backbone in `eval` durante il training; ha effetto solo con `FREEZE_BACKBONE=True`. `FLAG_RELU_INPLACE = True`: si tengono le ReLU in-place di torchvision (meno VRAM); `False` applica la patch. Nessun cambio automatico: se una tecnica richiede `False`, lo si dice in config. **Verificato in fase zero (check 2)** che con `True` tutte le tecniche a gradiente diano le stesse mappe del modello patchato. Con 5 epoche l'operatore ha misurato la stessa accuracy (0,9732) con ReLU in-place. | A · V (check 2) |

## Osservazioni sullo script esplorativo

| # | Osservazione | Esito |
|---|---|---|
| F1 | L'addestramento usava `model.train()` con pesi congelati: le statistiche BatchNorm del backbone cambiano. | Corretto: `enter_train_mode` con `FLAG_BATCHNORM_EVAL`. **Misurato dall'operatore: test 0,9732 con BatchNorm in `eval` contro ~0,949 senza.** L'effetto sull'accuracy è quindi a favore di `eval`. |
| F2 | L'accuracy citata era misurata sul test, la soglia è sulla validation. | Corretto: il notebook valuta la validation a ogni epoca; il check 6 misura sulla validation. |
| F3 | `torch.manual_seed(0)` mentre `SEED = 42`. | Corretto. |
| F4 | LR con `max_iter` 100 e avvisi soppressi. | Corretto (`LR_MAX_ITER`); restano filtrati i `FutureWarning`: vedi F9. |
| F5 | `torch.hub.load('pytorch/vision:v0.10.0', ...)` scarica codice da GitHub. | Corretto nella funzione di training (`models.densenet121(weights=...)`). Resta una cella con `torch.hub` più sotto (cella di esplorazione). |
| F6 | `random_split` non stratificato. | Corretto (`train_test_split` stratificato). Vedi F8 per l'uso dello split nella LR. |
| F7 | Preprocessing con PIL nel `DataLoader` (nessun worker): lento. | **Non adottato nello script esplorativo; decisione aperta.** Il check 6a verifica che il preprocessing su tensori sia equivalente (differenza massima misurata in `--quick`: 0,017 in unità normalizzate, soglia 0,05). Il check 6 misura il tempo per epoca con il preprocessing su GPU: da confrontare con quello del percorso PIL per decidere in tappa 1. |
| F8 (nuova) | In `run_train_lr` il `DataLoader` usa `trainval_dataset_lr` (60.000 immagini, validation inclusa), non `train_dataset_lr`: lo split viene calcolato e non usato. Inoltre `C`, solver e penalità della config non sono passati alla `LogisticRegression`. | Da correggere in tappa 1: la LR si addestra sui soli 54.000 del train; i parametri vengono dalla config. |
| F9 (nuova) | `warnings.filterwarnings('ignore', category=FutureWarning)` nasconde anche gli avvisi di sklearn sui parametri deprecati. | Il check 5 stampa il testo di ogni avviso non di convergenza. |
| F10 (nuova) | `set_batchnorm_eval(model)` applicato al modello intero è un no-op (non scende nei moduli); l'`enter_train_mode` successivo fa il lavoro giusto. Il learning rate è fisso a `1e-3` nel codice, non da `LEARNING_RATE_HEAD`. | Innocuo; si sistema in tappa 1. |
| F11 (nuova) | Nella copia della config dello script esplorativo `USE_SURROGATE` è ancora presente ("Obsolete"). | Nel `config.py` del repository non esiste più. |

## Verifiche già eseguite in sandbox (sklearn 1.8.0, dataset `digits` come sostituto)

- Con `penalty='l2'` esplicito sklearn 1.8 emette un `FutureWarning`: per L2 il parametro non viene passato (è il default in tutte le versioni).
- La somma dei coefficienti sulle classi è ≈ 1e-14 (conferma E02) e `decision_function == X·Wᵀ + b` esattamente. Da riconfermare su MNIST nel check 5.
