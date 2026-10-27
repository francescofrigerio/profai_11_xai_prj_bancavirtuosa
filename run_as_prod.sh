#!/bin/bash

# Valore di default: produzione
MODE="TRAIN"
NOPLOT=False
OUT_DIR="./outputs"

# Controlla se è stato passato l'argomento --demo o -d
if [[ "$1" == "--load" || "$1" == "--LOAD" ]]; then
    MODE="LOAD"
    echo "Avvio train in modalità: $MODE PER VELOCIZZARE L'ESECUZIONE"
else
    echo "Avvio train in modalità: $MODE"
fi

# Controlla se è stato passato l'argomento --plot o -d
if [[ "$2" == "--noplot" || "$2" == "--NOPLOT" ]]; then
    NOPLOT=True
    echo "Avvio train in modalità: $MODE PER VELOCIZZARE L'ESECUZIONE"
else
    echo "Avvio train con plot attivato : $NOPLOT"
fi

# 1. Pulizia checkpoint dedicati alla modalità attiva
rm -rf "${OUT_DIR}/checkpoint*"

# 2. Esecuzione del codice Python
cd src || exit
python -m run_main --mode "$MODE" --noplot "$NOPLOT" 2>&1 | tee run_${MODE}.log
cd ..

ls -lrt $OUT_DIR
