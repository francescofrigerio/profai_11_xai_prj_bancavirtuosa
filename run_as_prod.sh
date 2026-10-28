#!/bin/bash

# Valore di default: produzione
MODE="train"
OUT_DIR="./outputs"

# Controlla se è stato passato l'argomento --demo o -d
if [[ "$1" == "--load" ]]; then
    MODE="load"
    echo "Avvio train in modalità: $MODE PER VELOCIZZARE L'ESECUZIONE"
else
    echo "Avvio train in modalità: $MODE"
fi

# Dir di lavoro
cd src || exit

# Pulizia checkpoint dedicati alla modalità attiva
rm -rf "${OUT_DIR}/checkpoint*"

# Esecuzione del codice Python
python -m run_main --mode "$MODE" 2>&1 | tee run_${MODE}.log

# echo "ATTENZIONE AGLI ERRORI "
# pylint run_main.py | grep ": E"
# echo "ATTENZIONE AGLI ERRORI "

ls -lrt $OUT_DIR
