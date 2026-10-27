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

echo "Il codice non deve essere eseguito nel codespaces"
exit

# 1. Pulizia checkpoint dedicati alla modalità attiva
rm -rf "${OUT_DIR}/checkpoint*"

# 2. Esecuzione del codice Python
cd src || exit
python -m run_main --mode "$MODE" 2>&1 | tee run_${MODE}.log
cd ..

ls -lrt $OUT_DIR
