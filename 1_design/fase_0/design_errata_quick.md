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
| E01 | `MIN_VAL_ACCURACY` valeva 0,94 nella config e 0,95 nella tabella "da fissare" | 0,94 ovunque. È un avviso, non uno stop, e si misura sui pesi della migliore epoca. | A |
| E02 | Coefficienti LR relativi (globale) e assoluti (locale) | Il design resta: globale `w_c − media`, locale `w_a ⊙ x` esatta sul logit. Con L2 la somma dei coefficienti sulle classi è ≈ 0, quindi le due versioni coincidono. Aggiunto un test diagnostico. | A |
| E03 | "Riduzione a 28×28" ambigua | Tutte le mappe a 128×128 (bilineare se più piccole, es. Grad-CAM 4×4), poi media su area a 28×28. | P · V (check 12) |
| E04 | `config.py` previsto in tappa 1 ma necessario in tappa 0 | Anticipato in tappa 0, rifinito in tappa 1. | A |
| E05 | Snapshot del checkpoint (D08) incompleto | Aggiungere `VAL_SIZE`, `BATCH_SIZE`, `OPTIMIZER`, learning rate effettivo (e `NUM_EPOCHS` se utile). Un checkpoint con altri valori non è "compatibile". Il checkpoint di fase zero già salva questi campi. | P (tappa 1) |
| E06 | Test "senza GPU, dati minuscoli" contro pesi ImageNet | I test usano `weights=None` o un modello giocattolo. I pesi veri servono solo in fase zero e nel run reale. | A |
| E07 | Campione quantitativo "stratificato per classe" | Per classe **predetta**, N/10 per classe. | P |
| E08 | Composizione della galleria da 24 | 10 corrette (una per classe, a caso con seed, non scelte per bellezza) + 6 errori più confidenti + 8 dai quadranti (2 per quadrante, senza duplicati). | P |
| E09 | Maschera della tabella "fuori dalla cifra" | Maschera `digit` con dilatazione, indipendente da `MASK_MODE`. | P |
| E10 | Classe target del Grad-CAM di debug (D09) | Classe predetta, con l'etichetta vera nel titolo. | P |
| E11 | Arrotondamento del top-k (D22) | `k = ceil(TOPK_FRACTION · pixel nella maschera)`. | P |
| E12 | Aggregazione sui canali | IG e SHAP: somma. Occlusion: finestre su tutti e 3 i canali, poi media. | P · V (check 11) |
| E13 | Modalità `load` | Rete costruita con `weights=None` e caricamento dello state dict, senza scaricare ImageNet. | P |
| E14 | Testo narrativo e risultati dinamici | Si salva lo SHA-256 del checkpoint. Se non coincide con quello a cui si riferiscono le interpretazioni scritte a mano, il notebook mostra un banner visibile. | P (tappa 4) |
| E15 | `pip install` su Colab può forzare un riavvio del runtime | Il check 1 lo rileva; la cella 1 del notebook asserirà le versioni dopo l'installazione. Nessuna correzione automatica. | V (check 1) |
| E16 | Validazione della config incompleta | Estesa a nomi dei backend, intervalli, coerenza griglia/finestra, `N_QUANT` multiplo di `NUM_CLASSES`. Valore non valido → errore, nessun fallback. Implementata in `config.py`. | A |
| E17 | Dilatazione della maschera: elemento strutturante non specificato | Quadrato 3×3 (8-connesso), `MASK_DILATION` iterazioni. | P · V (check 8) |
| E18 | `SHAP_NUM_SAMPLES` mancava: i campioni di gradiente erano un numero magico | Aggiunto alla config (default 200, come shap). | P |

## Osservazioni sullo script esplorativo (fase0.py allegato)

| # | Osservazione | Conseguenza |
|---|---|---|
| F1 | L'addestramento usa `model.train()` con pesi congelati: le statistiche BatchNorm del backbone **cambiano** (il design le vuole in `eval`). L'accuracy 0,94–0,95 potrebbe dipendere da questo. | Il check 6 addestra entrambe le varianti e confronta accuracy e deriva delle statistiche. La scelta resta all'operatore. |
| F2 | L'accuracy citata è misurata sul test, mentre la soglia è sulla validation. | Il check 6 misura sulla validation. |
| F3 | `torch.manual_seed(0)` mentre `SEED = 42`. | Un unico `SEED` per Python, NumPy e torch. |
| F4 | LR con `max_iter` di default (100) e tutti gli avvisi soppressi: una mancata convergenza resterebbe invisibile. | Design: `LR_MAX_ITER=1000` e avvisi visibili (check 5). |
| F5 | `torch.hub.load('pytorch/vision:v0.10.0', ..., pretrained=True)` scarica il codice da GitHub. | Si usa `torchvision.models.densenet121` con l'enum dei pesi. |
| F6 | `random_split` non stratificato; gli split di LR e DenseNet coincidono solo perché il seed è lo stesso. | Split stratificato unico per indice (D-data). |
| F7 | Preprocessing con PIL nel `DataLoader` (nessun worker): lento. | Preprocessing su tensori, verificato equivalente (check 6a). |
| F8 | Nell'ultima cella `optimizer_adam` non è definito e `loss_function.item()` non esiste. | Il checkpoint finale salverà l'ottimizzatore corretto e la loss reale (tappa 1). |

## Verifiche già eseguite in sandbox (sklearn 1.8.0, dataset `digits` come sostituto)

- Con `penalty='l2'` esplicito sklearn 1.8 emette un `FutureWarning`: per L2 il parametro non viene passato (è il default in tutte le versioni).
- La somma dei coefficienti sulle classi è ≈ 1e-14 (conferma E02) e `decision_function == X·Wᵀ + b` esattamente. Da riconfermare su MNIST nel check 5.
