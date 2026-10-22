# Errata al design v1 – XAI Banca Virtuosa

Il notebook di design originale non viene modificato. Questo file raccoglie le correzioni e le
precisazioni approvate dall'operatore e quelle ancora da confermare. Stato: **A** = approvata (l'operatore ha approvato "tutte le modifiche indicate nel file errata", salvo F7), **V** = verificata in fase zero.

## Modifica richiesta

| ID | Modifica | Stato |
|---|---|---|
| D03-bis | Rimosso il modello surrogato. Spariscono `USE_SURROGATE` (config e tabella parametri §5), il "surrogato opzionale" in `baseline.py` (§4), il paragrafo "Surrogato" e la metrica di fedeltà come % di accordo (§8), il riferimento in §13 punto 8. Aggiunto alle esclusioni di §15. Restano: l'accordo tra le predizioni di DenseNet e LR nei quadranti (statistica descrittiva) e la fedeltà come deletion (D15). | A |

## Correzioni e precisazioni

| ID | Punto | Decisione | Stato |
|---|---|---|---|
| E01 | `MIN_VAL_ACCURACY` valeva 0,94 nella config e 0,95 nella tabella "da fissare" | **0,95** ovunque (decisione dell'operatore, dopo il run da 5 epoche con BatchNorm in `eval`: 0,9732 sul test). È un avviso, non uno stop, e si misura sull'accuracy di validation dell'**ultima** epoca (non c'è più selezione della migliore, vedi E20). | A |
| E02 | Coefficienti LR relativi (globale) e assoluti (locale) | Il design resta: globale `w_c − media`, locale `w_a ⊙ x` esatta sul logit. Con L2 la somma dei coefficienti sulle classi è ≈ 0, quindi le due versioni coincidono. Aggiunto un test diagnostico. | A |
| E03 | "Riduzione a 28×28" ambigua | Tutte le mappe a 128×128 (bilineare se più piccole, es. Grad-CAM 4×4), poi media su area a 28×28. | A · confermato (check 12, figura 12) |
| E04 | `config.py` previsto in tappa 1 ma necessario in tappa 0 | Anticipato in tappa 0, rifinito in tappa 1. | A |
| E05 | Snapshot del checkpoint (D08) incompleto | Aggiungere `VAL_SIZE`, `BATCH_SIZE`, `NUM_EPOCHS`, `OPTIMIZER`, learning rate effettivo e `FLAG_BATCHNORM_EVAL` (cambia i pesi). `FLAG_RELU_INPLACE` no: non cambia i pesi. Un checkpoint con altri valori non è "compatibile". Il checkpoint di fase zero già salva questi campi. | A (da implementare in tappa 1) |
| E06 | Test "senza GPU, dati minuscoli" contro pesi ImageNet | I test usano `weights=None` o un modello giocattolo. I pesi veri servono solo in fase zero e nel run reale. | A |
| E07 | Campione quantitativo "stratificato per classe" | Per classe **predetta**, N/10 per classe. | A |
| E08 | Composizione della galleria da 24 | 10 corrette (una per classe, a caso con seed, non scelte per bellezza) + 6 errori più confidenti + 8 dai quadranti (2 per quadrante, senza duplicati). | A |
| E09 | Maschera della tabella "fuori dalla cifra" | Maschera `digit` con dilatazione, indipendente da `MASK_MODE`. | A |
| E10 | Classe target del Grad-CAM di debug (D09) | Classe predetta, con l'etichetta vera nel titolo. | A |
| E11 | Arrotondamento del top-k (D22) | `k = ceil(TOPK_FRACTION · pixel nella maschera)`. | A |
| E12 | Aggregazione sui canali | IG e SHAP: somma. Occlusion: finestre su tutti e 3 i canali, poi media. **Verificato in `--quick` (check 11): le finestre sovrapposte sono mediate da Captum, le mappe dei 3 canali sono identiche, 226 forward = 225 finestre + l'originale.** | A |
| E13 | Modalità `load` | Rete costruita con `weights=None` e caricamento dello state dict, senza scaricare ImageNet. | A |
| E14 | Testo narrativo e risultati dinamici | Si salva lo SHA-256 del checkpoint. Se non coincide con quello a cui si riferiscono le interpretazioni scritte a mano, il notebook mostra un banner visibile. | A (da implementare in tappa 4) |
| E15 | `pip install` su Colab può forzare un riavvio del runtime | Il check 1 lo rileva; la cella 1 del notebook asserirà le versioni dopo l'installazione. Nessuna correzione automatica. | V (check 1) |
| E16 | Validazione della config incompleta | Estesa a nomi dei backend, intervalli, coerenza griglia/finestra, `N_QUANT` multiplo di `NUM_CLASSES`. Valore non valido → errore, nessun fallback. Implementata in `config.py`. | A |
| E17 | Dilatazione della maschera: elemento strutturante non specificato | Quadrato 3×3 (8-connesso), `MASK_DILATION` iterazioni. | A · confermato (check 8, figura 8) |
| E18 | `SHAP_NUM_SAMPLES` mancava: i campioni di gradiente erano un numero magico | Aggiunto alla config (default 200, come shap). | A |
| E19 | `IMAGE_SIZE` indicava due cose (MNIST 28×28 e input DenseNet 128×128) | Due parametri: `IMAGE_SIZE = 28` (MNIST, input della LR) e `DENSENET_SIZE = 128`. Le validazioni su griglia LIME, finestra Occlusion e stride 32 usano `DENSENET_SIZE`. Preso dal config dello script esplorativo dell'operatore. | A |
| E20 | Early stopping (D07) | **Eliminato** (con 5 epoche complica il codice senza vantaggio). Via `EARLY_STOPPING_PATIENCE`, via il relativo test e la sua menzione in §7. Numero di epoche fisso; si salvano i pesi dell'**ultima** epoca. | A |
| E21 | Due nuovi flag in config | `FLAG_BATCHNORM_EVAL = True`: BatchNorm del backbone in `eval` durante il training; ha effetto solo con `FREEZE_BACKBONE=True`. `FLAG_RELU_INPLACE = True`: si tengono le ReLU in-place di torchvision (meno VRAM); `False` applica la patch. Nessun cambio automatico: se una tecnica richiede `False`, lo si dice in config. **Confermato dall'operatore: entrambi i flag `True` nel CONFIG** (`FLAG_RELU_INPLACE=True` significa "tenere le ReLU in-place"). **Verificato in fase zero (check 2):** con ReLU in-place e con ReLU patchate Grad-CAM, Integrated Gradients, Saliency e Gradient SHAP danno la stessa mappa (differenza relativa ≤ 1e-6, Grad-CAM esattamente 0). Il primo FAIL di Gradient SHAP (Captum) dipendeva da un errore dello script (generatore NumPy non seminato). Con 5 epoche l'operatore ha misurato la stessa accuracy (0,9732) con ReLU in-place. | A · verificato (check 2) |
| E22 | Saliency (Captum) come sesta tecnica XAI, richiesta dall'operatore | **Approvata dall'operatore, revocabile**: se nel report complicasse le spiegazioni si toglie. Modifica la decisione chiusa sull'insieme delle tecniche (§3): è D-bis. Effetti: nuovo `xai/saliency.py`; matrici di accordo 7×7 invece di 6×6; una colonna in più nella galleria; un caso in più nei test del contratto e nel test del segnale piantato. Costo a runtime trascurabile (un solo backward). `abs=True` e somma dei canali, per coerenza con IG e SHAP. Il check 13 misura tempo e quanto dell'attribuzione cade sulla cifra. **Misurato sul modello addestrato: 0,049 s per immagine; solo il 23 % dell'attribuzione cade sulla cifra contro il 22 % di area (per immagine: 25 %, 19 %, 26 % contro 25 %, 27 %, 15 %): la mappa di Saliency si disperde su tutto il quadro, non è guidata dalla cifra.** | A · misurato (check 13) |
| E23 | `IG_STEPS = 50` non basta | Misurato sul modello addestrato: errore massimo di completezza 20,3 % a 50 passi, 5,1 % a 100, **3,0 % a 200**, 1,6 % a 400 (soglia 5 %). Cala in modo monotono, quindi converge. `IG_STEPS = 200` (0,53 s per immagine). Impostato nel `config.py`, in attesa di veto. | A (proposta della fase zero, veto possibile) |
| E24 | Backend alternativi di LIME e SHAP (Captum) | **Eliminati** (decisione dell'operatore, condivisa): LIME e SHAP usano sempre i pacchetti di riferimento `lime` e `shap`. Via `LIME_BACKEND` e `SHAP_BACKEND` dalla config, via `xai/lime_captum.py` e `xai/shap_captum.py`, via i relativi test. Captum resta per Grad-CAM, IG, Occlusion e Saliency. Nel check 3 entrambi i pacchetti funzionano, sono deterministici e rispettano il limite di tempo (lime 3,1 s, shap 6,5 s per immagine). Costo: SHAP con il pacchetto pesa 6,5 s su ~10,6 s per immagine; se serve tempo si agisce su `SHAP_NUM_SAMPLES` o `N_QUANT`, non sul backend. | A |
| E25 | I check 3, 4, 13 giravano su una testa **casuale** senza dirlo | Il checkpoint di fase zero vive solo nella sessione Colab (non è in git): in una sessione nuova `get_model` usava una testa casuale e stampava solo una nota. Ora i check 3, 4, 13, 14 **rifiutano** di girare senza il checkpoint (salvo `--allow-random-head`), e la figura del check 3 riporta "RANDOM HEAD" in rosso. Le conclusioni tratte dalla figura 03 e dai numeri di check 3/4/13 delle esecuzioni senza checkpoint non sono valide. | A |

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
| 03 backend e 03 immagine | **Ritirate le osservazioni precedenti** (mappa `lime/captum` vuota, 7 classificato 8, accordo tra backend SHAP): la figura veniva da una testa casuale (E25). Con testa casuale il 7 predetto 8 e la mappa vuota sono attesi. Resta valido solo che `lime/lime` e `shap/shap` girano, sono deterministici e di forma corretta. | Da rifare con il checkpoint addestrato: `--checks 6,14,3,4,13 --no-bn-comparison`. |

## Esito del secondo giro (`--checks 2,3,4,13,14`, sessione nuova, nessun checkpoint)

| Check | Esito | Valore |
|---|---|---|
| 2 ReLU in-place | PASS | Tutte le tecniche a gradiente identiche in-place/patchate. **Valido** (confronto a pesi identici). |
| 3 LIME/SHAP | tempi validi, mappe no | lime 3,1 s, shap 6,5 s; `shap/captum` ora deterministico (era il seed NumPy). Mappe: testa casuale. |
| 4 IG | non conclusivo | Errore di completezza 22,9 % (50 passi), 14,6 % (100), 4,5 % (200), 4,4 % (400). Il plateau a ~4,4 % con una testa casuale (differenze di logit minime, cancellazioni in float32) non permette di scegliere `IG_STEPS`. |
| 13 Saliency | non conclusivo | 22,7 % dell'attribuzione sulla cifra contro 22,4 % di area: testa casuale. Nota: IG vale 100 % **per costruzione** (attribuzione nulla dove l'immagine coincide con la base nera), quindi non è un confronto equo. |
| 14 Round trip | non eseguibile | Nessun checkpoint nella sessione. |

## Esito del terzo giro (`--checks 6,14,3,4,13 --no-bn-comparison`, modello addestrato)

Valido: il check 14 dimostra che il checkpoint ricaricato riproduce l'accuracy del training (0,9700 contro 0,9700) e che i tre campioni sono classificati correttamente con ~100 % di confidenza. Il training è riproducibile: le accuracy per epoca sono identiche a quelle del run precedente su una sessione diversa.

| Check | Esito | Valore |
|---|---|---|
| 6 Addestramento | PASS | 0,9497 → 0,9700 in 5 epoche; 60 s + 7 s per epoca; BatchNorm del backbone invariate. |
| 14 Round trip | PASS | Vedi sopra. |
| 3 LIME/SHAP | PASS | lime 3,3 s, shap 6,9 s per immagine, entrambi deterministici. Figura con il modello addestrato: LIME evidenzia lo stelo e l'estremo della barra del 7 (celle rosse lungo la diagonale); SHAP è a grana fine, con rumore diffuso e un picco alla base dello stelo. |
| 4 IG | FAIL a 50 passi | Vedi E23: converge. |
| 13 Saliency | WARN | Vedi E22. |

## Cautele interpretative per il report (E26)

- **Integrated Gradients vale 100 % "sulla cifra" per costruzione.** Con base nera l'attribuzione è nulla dove l'immagine è nera: la tabella dell'attribuzione fuori dalla cifra darà 0 % per IG a prescindere dal modello. Non va letta come un merito rispetto alle altre tecniche.
- **SHAP con sfondi presi dal dataset** attribuisce anche a pixel dove l'immagine spiegata è vuota ma gli sfondi hanno inchiostro (l'assenza di tratto conta). È legittimo ma va spiegato a un pubblico non tecnico.
- **Il rumore di LIME e SHAP da un seed all'altro** limita l'accordo massimo raggiungibile con qualunque altra tecnica: il nuovo check 15 lo misura.

## Chiusura della tappa 0

Run finale completo (Colab, Tesla T4, Python 3.13, torch 2.11.0+cu130, torchvision 0.26.0, numpy 2.1.3, scipy 1.16.3, scikit-learn 1.6.1, captum 0.9.0, lime 0.2.0.1, **shap 0.54.1**). `phase0_results.json` contiene i 15 check, nessun traceback, nessuna esecuzione `--quick`. `phase0_checks.py` e `config.py` sono considerati definitivi per la fase zero.

### Valori fissati

| Parametro | Valore | Evidenza | Stato |
|---|---|---|---|
| LIME / SHAP | pacchetti `lime` e `shap` | check 3: 3,2 s e 6,8 s per immagine, deterministici | A (E24) |
| `NUM_EPOCHS` | 5 | validation 0,9497 → 0,9700; la loss cala ancora dell'11,6 % ma l'accuracy è già 0,97 | A (confermato dall'operatore) |
| `MIN_VAL_ACCURACY` | 0,95 | ultima epoca 0,9700 | A |
| `FLAG_BATCHNORM_EVAL` / `FLAG_RELU_INPLACE` | `True` / `True` | 0,9700 contro 0,9493 senza BatchNorm in `eval`; mappe identiche con ReLU in-place per Grad-CAM, IG, Saliency e SHAP | A · verificato |
| `BATCH_SIZE`, `LEARNING_RATE_HEAD` | 64, 1e-3 | 59 s + 7 s per epoca | A |
| `LR_C`, `LR_MAX_ITER` | 1,0, 1000 | validation 0,9222, convergenza in 248 iterazioni | A (figura delle mappe globali da rivedere nel notebook) |
| `IG_STEPS` | 200 | errore di completezza 3,0 % (50 passi: 20,3 %) | A |
| `OCCLUSION_WINDOW` / `OCCLUSION_STRIDE` | 16 / 8 | 226 forward, finestre sovrapposte mediate | A |
| `LIME_GRID_CELLS` | 8 | figura 7: 12–27 celle su 64 toccano la cifra; quickshift frammenta lo sfondo | A |
| `MASK_THRESHOLD` / `MASK_DILATION` | 0,1 / 1 | figura 8; maschera ~30 % dei pixel; mai vuota su 200 immagini | A |
| Regola di ricampionamento | bilineare → 128 → media su area | allineamento 100/100 e 16/16; 187 valori distinti contro 11 con nearest | A |
| `N_QUANT` | 100 | ~18 min (budget 20) | A |
| Saliency | sesta tecnica | 0,04 s per immagine; 23 % dell'attribuzione sulla cifra contro 22 % di area | A (revocabile) |
| `SHAP_NUM_SAMPLES` | **da decidere**: 200 oppure 100 | stabilità seed-to-seed (Spearman nella maschera): 0,80 a 200, 0,78 a 100 campioni; tempo per immagine dimezzato | aperto |

### Altri risultati utili per il report

- **Divario DenseNet–LR sulla validation: 4,78 punti** (0,9700 contro 0,9222), dentro la forchetta 4–6 attesa dal design.
- **Tetto di accordo tra tecniche.** LIME ha stabilità 0,85 e SHAP 0,80 (seed-to-seed): nessun confronto che le coinvolga potrà superare di molto quei valori. Da dichiarare nel report accanto alle matrici.
- **Riproducibilità.** Il training dà accuracy per epoca identiche in sessioni Colab diverse; le figure 8, 12 e 3 (con modello addestrato) sono identiche pixel per pixel tra due sessioni.
- **Tempi stimati di "Esegui tutto":** ~28 minuti in `load`, ~34 in `train` (limiti 30 e 45). Con `SHAP_NUM_SAMPLES=100` il calcolo sui numeri misurati dà circa 20 e 26 minuti.
- **Instabilità dell'ambiente osservata:** `shap` è passato da 0,52 a 0,54.1 tra due giri senza che nulla cambiasse nei risultati o nei tempi, ma una pin è necessaria. Avviso di SciPy: le opzioni `disp`/`iprint` di L-BFGS-B sono deprecate e spariranno in SciPy 1.18, con scikit-learn 1.6.1 il fit della LR potrebbe smettere di funzionare su un Colab futuro.

### Lasciato alla tappa 1

E05 (snapshot completo del checkpoint), E07–E11, E13, E14 (da implementare dove indicato); F8 (la LR si addestra sui soli 54.000 del train, con i parametri dalla config), F10; F7 (preprocessing su tensori o PIL) aperta: il percorso su tensori è equivalente (differenza massima 0,017 in unità normalizzate) e costa 59 s per epoca. La figura delle mappe globali della LR (check 5) non è stata ancora esaminata: il notebook la rigenererà.
