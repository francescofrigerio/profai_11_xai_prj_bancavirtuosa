#!/bin/bash

# Interrompe lo script al primo errore
set -e

# Funzione per i controlli ordinari (GPU, Python, ecc.)
run_help() {
    echo "----------------------------------------------------"
    echo "HELP IN LINEA :  cd src + PYTHONPATH=. python --help"
    echo "1 -> setup.sh --init  (inizializza progetto , install. librerie/venv) "
    # echo "2 -> setup.sh --change  (modifica struttura progetto , installazione nuove librerie) "
    echo "2 -> setup.sh --install  (installazione librerie tramite requirements.txt) "
    echo "3 -> setup.sh --checks  (controllo GPU/CPU verifica installazione librerie) "
    echo "4 -> source .venv/bin/activate per attivare l'ambiente virtuale"
    echo "5 -> deactivate per disattivare l'ambiente virtuale"
    echo "6 -> ruff check file.py controllo superficiale di un file python"
    # echo "7 -> cd src + PYTHONPATH=. pylint train/train_baseline.py per controllo profondo di un file python"
    
    echo "Attenzione!! comandi per pulire ambiente e ricrearlo (da eseguire solo se necessario)"
    echo "pip cache purge"
    echo "deactivate"
    echo "rm -rf .venv"
    echo "python3 -m venv .venv"
    echo "source .venv/bin/activate"
    echo "pip install --upgrade pip"
    # torch installed on colab
    # echo "pip install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir"
    echo "pip install -r requirements.txt --no-cache-dir"

    echo "----------------------------------------------------"
}

run_checks() {
    echo "----------------------------------------------------"
    echo "Verifica Ambiente e Hardware "
    if [ -d ".venv" ]; then
        echo "Visualizzazione pacchetti installati:"
        .venv/bin/pip list | grep -E "torch|captum|lime|shap|pytest|pylint|ruff" || true
        echo "----------------------------------------------------"
        echo "Verifica GPU/CPU via PyTorch:"
        .venv/bin/python -c "import torch; print('GPU Disponibile:', torch.cuda.is_available()); print('Nome GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Nessuna GPU. Uso CPU')"
        
        echo "Verifica compatibilita' tra torch e torchvision:"
        echo "torchvision non e' utilizzato in NLP ma viene richiesto che sia installato da torch"
        .venv/bin/python -c "import torchvision; print(torchvision.__version__)"
        .venv/bin/python -c "import torch; print(torch.__version__)"

    else
        echo "[ERRORE] Ambiente virtuale .venv non trovato. Lancia lo script con il flag --init"
        exit 1
    fi
    echo "----------------------------------------------------"
    echo "Per iniziare a lavorare, attiva l'ambiente nel tuo terminale con:"
    echo "source .venv/bin/activate"
    echo "----------------------------------------------------"
}

echo "----------------------------------------------------"
echo "1. Verifica Python (max versione 3.12)"
python3 --version

# Controllo del parametro in ingresso
if [ "$1" == "--init" ]; then
    echo "----------------------------------------------------"
    echo "INIZIALIZZAZIONE PROGETTO "
    
    # Crea file iniziali nella root se non esistono
    touch .gitignore README.md 
    echo -e ".venv/\n__pycache__/\n*.pyc\nmodel/\n.ipynb_checkpoints/" > .gitignore

    echo "2. Creazione della Struttura delle Directory e File"

     # Cartella src/ e relative sottocartelle/file
     # codice per ciclo ML training test evaluate
    mkdir -p src/
    touch src/logger.py   # script logger debug,info
    touch src/config.py   # script configuration
    touch src/data.py   # script con process data
    touch src/baseline.py   # script con baseline models
    touch src/model.py   # script model
    touch src/train.py   # script con training models
    touch src/metrics.py # validazione delle metriche 
    touch src/run_main.py   # script main
    
    # Cartella tests/ e sottocartelle
    mkdir -p tests/LR tests/CNN
    touch tests/LR/.gitkeep
    touch tests/CNN/.gitkeep

    # Cartella model/ e file
    mkdir -p model
    # touch model/model.pkl
    
     mkdir -p logs
     echo "*" >./logs/.gitignore
     echo "!.gitignore" >>./logs/.gitignore

     mkdir -p architecture notebooks
     touch architecture/.gitkeep
     touch notebooks/.gitkeep

    # Cartella docs/ e relativi file markdown
    mkdir -p docs
    touch docs/01-architecture.md
    touch docs/02-environment.md
    touch docs/03-training.md
    touch docs/04-test.md
    
    # Cartella links/ e relativi file
    mkdir -p links
    touch links/.gitkeep

    echo "Struttura del progetto creata con successo."

    echo "3. Creazione Ambiente Virtuale"
    # Crea l'ambiente virtuale se non esiste già
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
        echo "Ambiente virtuale creato con successo."
    else
        echo "L'ambiente virtuale esiste già."
    fi

    source .venv/bin/activate
    
    echo "4. Installazione Pacchetti"
    # Aggiorna pip all'interno dell'ambiente virtuale
    .venv/bin/pip install --upgrade pip

    # Installa le librerie richieste nel virtualenv
     echo "INSTALLAZIONE LIBRERIE PROGETTO "
    # Per installare le lib dal file requirements nell'ambiente virtuale
    .venv/bin/pip install -r requirements.txt
    
    echo "Elenco pacchetti installati"
    .venv/bin/pip list

    echo "----------------------------------------------------"
    echo " Setup completato con successo!"
    echo " Controlli sul codespaces"
    echo " nvidia-smi"
    echo " Per iniziare a lavorare, attiva l'ambiente virtuale con:"
    echo " source .venv/bin/activate"
    echo " start verifica GPU/CPU"
    # echo " python -c 'import torch; print('GPU Disponibile:', torch.cuda.is_available()); print('Nome GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Nessuna Gpu.Usa CPU')"
    echo ' python -c "import torch; print('\''GPU Disponibile:'\'', torch.cuda.is_available()); print('\''Nome GPU:'\'', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '\''Nessuna GPU. Usa CPU'\'')"'
    echo " end verifica GPU/CPU"
    echo "----------------------------------------------------"
    
    # controllo GPU e stato ambiente
    run_checks
    echo "----------------------------------------------------"
    
elif [ "$1" == "--change" ]; then
    echo "----------------------------------------------------"
    echo "MODIFICA STRUTTURA PROGETTO "
    echo "----------------------------------------------------"
elif [ "$1" == "--install" ]; then
    echo "In caso servisse ripartire da zero eseguire"
    echo "python3 -m venv .venv"
    echo "source .venv/bin/activate"
    echo ".venv/bin/pip install --upgrade pip"
    echo "----------------------------------------------------"
    echo "INSTALLAZIONE LIBRERIE PROGETTO "
    # Per installare le lib dal file requirements nell'ambiente virtuale
    .venv/bin/pip install -r requirements.txt
    echo "----------------------------------------------------"
elif [ "$1" == "--checks" ]; then
    run_checks
else
    # controllo GPU e stato ambiente
    run_help
fi

