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
| E21 | Due nuovi flag in config | `FLAG_BATCHNORM_EVAL = True`: BatchNorm del backbone in `eval` durante il training; ha effetto solo con `FREEZE_BACKBONE=True`. `FLAG_RELU_INPLACE = True`: si tengono le ReLU in-place di torchvision (meno VRAM); `False` applica la patch. Nessun cambio automatico: se una tecnica richiede `False`, lo si dice in config. **Confermato dall'operatore: entrambi i flag `True` nel CONFIG** (`FLAG_RELU_INPLACE=True` significa "tenere le ReLU in-place"). **Da verificare in fase zero (check 2):** nel primo run Grad-CAM, IG e SHAP (shap) danno le stesse mappe con ReLU in-place e patchate; Gradient SHAP (captum) risultava diverso, ma l'esito era inquinato da un errore dello script (generatore NumPy non seminato), da rieseguire. Con 5 epoche l'operatore ha misurato la stessa accuracy (0,9732) con ReLU in-place. | A · V (check 2, da rieseguire dopo la correzione del seed NumPy) |
| E22 | Saliency (Captum) come sesta tecnica XAI, richiesta dall'operatore | **Approvata dall'operatore, revocabile**: se nel report complicasse le spiegazioni si toglie. Modifica la decisione chiusa sull'insieme delle tecniche (§3): è D-bis. Effetti: nuovo `xai/saliency.py`; matrici di accordo 7×7 invece di 6×6; una colonna in più nella galleria; un caso in più nei test del contratto e nel test del segnale piantato. Costo a runtime trascurabile (un solo backward). `abs=True` e somma dei canali, per coerenza con IG e SHAP. Il check 13 misura tempo e quanto dell'attribuzione cade sulla cifra. | A · V (check 13) |
| E23 | `IG_STEPS = 50` non basta | Nel run completo la completezza di IG ha errore relativo massimo 20,3 % a 50 passi (soglia 5 %). Il check 4 ora fa uno sweep (50/100/200/400) e propone il più piccolo che rispetta la soglia. IG usa `internal_batch_size=50` per non saturare la VRAM. | V (check 4) |

## Osservazioni sullo script esplorativo

| # | Osservazione | Esito |
|---|---|---|
| F1 | L'addestramento usava `model.train()` con pesi congelati: le statistiche BatchNorm del backbone cambiano. | Corretto: `enter_train_mode` con `FLAG_BATCHNORM_EVAL`. **Misurato dall'operatore: test 0,9732 con BatchNorm in `eval` contro ~0,949 senza.** L'effetto sull'accuracy è quindi a favore di `eval`. |
| F2 | L'accuracy citata era misurata sul test, la soglia è sulla validation. | Corretto: il notebook valuta la validation a ogni epoca; il check 6 misura sulla validation. |
| F3 | `torch.manual_seed(0)` mentre `SEED = 42`. | Corretto. |
| F4 | LR con `max_iter` 100 e avvisi soppressi. | Corretto (`LR_MAX_ITER`); restano filtrati i `FutureWarning`: vedi F9. |
| F5 | `torch.hub.load('pytorch/vision:v0.10.0', ...)` scarica codice da GitHub. | Corretto nella funzione di training (`models.densenet121(weights=...)`). Resta una cella con `torch.hub` più sotto (cella di esplorazione). |
| F6 | `random_split` non stratificato. | Corretto (`train_test_split` stratificato). Vedi F8 per l'uso dello split nella LR. |
| F7 | Preprocessing con PIL nel `DataLoader` (nessun worker): lento. | **Non adottato nello script esplorativo; decisione aperta.** Il check 6a verifica che il preprocessing su tensori sia equivalente (differenza massima misurata in `--quick`: 0,017 in unità normalizzate, soglia 0,05). Misurato nel run completo: 60 s di training + 8 s di validation per epoca con il preprocessing su GPU (65 s con BatchNorm in train). Da confrontare con il tempo del percorso PIL per decidere in tappa 1. |
| F8 (nuova) | In `run_train_lr` il `DataLoader` usa `trainval_dataset_lr` (60.000 immagini, validation inclusa), non `train_dataset_lr`: lo split viene calcolato e non usato. Inoltre `C`, solver e penalità della config non sono passati alla `LogisticRegression`. | Da correggere in tappa 1: la LR si addestra sui soli 54.000 del train; i parametri vengono dalla config. |
| F9 (nuova) | `warnings.filterwarnings('ignore', category=FutureWarning)` nasconde anche gli avvisi di sklearn sui parametri deprecati. | Il check 5 stampa il testo di ogni avviso non di convergenza. |
| F10 (nuova) | `set_batchnorm_eval(model)` applicato al modello intero è un no-op (non scende nei moduli); l'`enter_train_mode` successivo fa il lavoro giusto. Il learning rate è fisso a `1e-3` nel codice, non da `LEARNING_RATE_HEAD`. | Innocuo; si sistema in tappa 1. |
| F11 (nuova) | Nella copia della config dello script esplorativo `USE_SURROGATE` è ancora presente ("Obsolete"). | Nel `config.py` del repository non esiste più. |

## Verifiche già eseguite in sandbox (sklearn 1.8.0, dataset `digits` come sostituto)

- Con `penalty='l2'` esplicito sklearn 1.8 emette un `FutureWarning`: per L2 il parametro non viene passato (è il default in tutte le versioni).
- La somma dei coefficienti sulle classi è ≈ 1e-14 (conferma E02) e `decision_function == X·Wᵀ + b` esattamente. Da riconfermare su MNIST nel check 5.

## Esiti del run completo di fase zero (Colab, T4, torch 2.11, sklearn 1.6.1)

| Check | Esito | Nota |
|---|---|---|
| 1 Ambiente | PASS | Nessun riavvio necessario. Avviso `scipy.optimize ... iprint` deprecato (rimosso in SciPy 1.18): rischio futuro per sklearn 1.6.1. |
| 2 ReLU in-place | FAIL (da rieseguire) | Grad-CAM, IG e SHAP (shap) identici in-place e patchati. Gradient SHAP (captum) diverso (0,56), ma lo stesso confronto con modello identico dava 0,64 nel check 3: causa NumPy non seminato. |
| 3 Backend | WARN | lime/lime 3,0 s, lime/captum 1,6 s, shap/shap 6,9 s, shap/captum 0,5 s. `shap/captum` non deterministico (stessa causa). |
| 4 IG | FAIL | Completezza 20,3 % a 50 passi: sweep nel check 4. |
| 5 LR | REVIEW | Accuracy validation 0,9222 (attesa ~0,92); convergenza in 248 iterazioni; somma coefficienti ≈ 1e-13; logit esatto; 62,6 s di fit. |
| 6 Addestramento | PASS | Ultima epoca 0,9700 con BatchNorm in `eval` (0,9493 con BatchNorm in train, deriva delle statistiche 31,5). Loss di validation ancora in calo dell'11,6 % all'ultima epoca. |
| 9 Tempi | PASS | `N_QUANT=100` ≈ 17,3 min; "Esegui tutto" ≈ 27 min (load) / 32 min (train). SHAP (shap) è 6,9 s su 10,4 s per immagine. |

## Note dalle figure del run completo (da confermare)

| Figura | Osservazione | Raccomandazione |
|---|---|---|
| 07 griglia LIME | La griglia 8×8 spezza ogni cifra in 12–27 celle su 64, con celle di larghezza paragonabile al tratto. Quickshift produce 100–130 segmenti minuscoli che frammentano anche lo sfondo: è chiaramente peggiore (conferma D11). | Confermare `LIME_GRID_CELLS = 8`. |
| 08 maschera | A soglia 0,1 senza dilatazione la maschera segue il tratto con il suo alone antialias (17 % dei pixel). Con dilatazione 1 diventa il 30 %, i tratti ingrassano e i piccoli anelli di 8, 6, 9 si riducono. Le soglie 0,05–0,3 cambiano l'area solo di 1–3 punti. | Confermare `MASK_THRESHOLD = 0,1` e `MASK_DILATION = 1` (alone utile con mappe grossolane come Grad-CAM e LIME); tenere `0` come alternativa documentata. |
| 12 ricampionamento | Grad-CAM nativo 4×4 a blocchi; la regola bilineare→128→area dà una mappa liscia e fedele, nettamente migliore del nearest. Piccole ondulazioni sui bordi sinistro e inferiore, innocue. | Confermare E03. |
| 03 backend | `lime/lime`: mappa a celle, plausibile. `lime/captum`: **mappa vuota** su questa immagine (il Lasso predefinito azzera tutto). `shap/shap` e `shap/captum`: mappe quasi identiche tra loro, a pixel, con picchi sugli estremi del tratto: i due backend concordano. | Tenere `LIME_BACKEND = lime`. Un backend che restituisce mappe vuote ora risulta non selezionabile. |
| 03 immagine | Il primo 7 del test viene classificato **8**. Con il 97 % di accuracy è un evento da ~3 %, plausibile ma da verificare. | Il nuovo check 14 ricarica il checkpoint e confronta l'accuracy di validation con quella del training. |
