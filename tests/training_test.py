
import sys
import os
from pathlib import Path
from sklearn.metrics import accuracy_score

# Trova la cartella 'src' partendo dalla posizione del file di test corrente
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from training import get_evaluate_model , read_checkpoint , build_densenet
from config import CONFIG
from logger import LOGGER

def test_load_checkpoint(cfg=CONFIG, logger=LOGGER):
    """
        Verifica che il checkpoint esista e che il modello si carichi correttamente.
    """
    ROOT_DIR = Path(__file__).resolve().parent.parent
    print("CWD corrente:", os.getcwd())
    
    print("CWD corrente:", ROOT_DIR)
    print("Percorso assoluto del checkpoint:", Path(cfg.CHECKPOINT_PATH).resolve())
    
    path = Path(cfg.CHECKPOINT_PATH)
    assert path.exists(), f"Checkpoint non trovato al percorso: {path}"
    
    # Lettura sicura del checkpoint
    saved = read_checkpoint(path)
    
    # Gestione robusta della state_dict (compatibile con dizionario o pesi diretti)
    state_dict = saved["state_dict"] if isinstance(saved, dict) and "state_dict" in saved else saved
    
    # Istanziazione e caricamento pesi
    model = build_densenet(cfg, logger)
    model.load_state_dict(state_dict)
    
    # Asserzioni di controllo
    assert model is not None, "Il modello non è stato instanziato correttamente."
    logger.info("[TEST] Checkpoint e modello caricati con successo.")



def test_model_evaluation(cfg, logger, test_loader, device_):
    """
        Verifica l'esecuzione della pipeline di valutazione e la coerenza degli output.
    """
    
    # Esegue la funzione di valutazione esistente
    X_test, y_true, y_pred = get_evaluate_model(cfg=cfg,
                                                logger=logger,
                                                pretrained_=False,  # In fase di evaluation solitamente usiamo il checkpoint caricato
                                                test_loader_=test_loader,
                                                device_=device_
                                            )
    
    # Verifiche strutturali sui dati restituiti
    assert len(X_test) > 0, "Il dataset di test restituito è vuoto."
    assert len(y_true) == len(y_pred), "Il numero di etichette reali e predette non coincide."
    
    # Calcolo metrica di controllo
    acc = accuracy_score(y_true, y_pred)
    logger.info(f"[TEST] Accuratezza misurata sul test set: {acc:.4f}")
    
    # Assicuriamoci che l'accuracy sia un valore percentuale valido (tra 0 e 1)
    assert 0.0 <= acc <= 1.0, f"Valore di accuracy non valido: {acc}"
