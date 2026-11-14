#!/bin/bash

# Funzione per mostrare l'utilizzo dello script
usage() {
    echo "Uso: run_lint.sh <nome_file.py> [-e] [-v pattern]"
    echo "  -e          Mostra solo gli errori critici (: E)"
    echo "  -v PATTERN  Esclude dal report le righe contenenti il pattern specificato (es. 'trailing')"
    exit 1
}

# Verifica che sia stato passato almeno un argomento
if [ $# -eq 0 ]; then
    usage
fi

TARGET_INPUT=""
ERRORS_ONLY=false
EXCLUDE_PATTERN=""

# Parsing degli argomenti
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -e|--errors) 
            ERRORS_ONLY=true 
            shift 
            ;;
        -v|--exclude) 
            EXCLUDE_PATTERN="$2"
            shift 2 
            ;;
        -h|--help) 
            usage 
            ;;
        *) 
            TARGET_INPUT="$1"
            shift 
            ;;
    esac
done

if [ -z "$TARGET_INPUT" ]; then
    echo "Errore: Devi specificare il nome del file Python."
    usage
fi

TARGET_FILE="$TARGET_INPUT"

# Se il file non esiste nel percorso esatto, cercalo esclusivamente dentro 'src'
if [ ! -f "$TARGET_FILE" ]; then
    if [ ! -d "src" ]; then
        echo "Errore: Cartella 'src' non trovata nella directory corrente."
        exit 1
    fi

    # Cerca solo dentro la cartella src ricorsivamente
    # echo CONTROLLO DOVE SONO
    # pwd 
    # ls src | wc -l
    # ls src/$TARGET_INPUT 
    FOUND_FILE=$(find src -type f -name "$TARGET_INPUT" | head -n 1)
    
    if [ -n "$FOUND_FILE" ]; then
        TARGET_FILE="$FOUND_FILE"
        echo "File trovato in: $TARGET_FILE"
    else
        echo "Errore: Impossibile trovare il file '$TARGET_INPUT' all'interno di 'src'."
        exit 1
    fi
fi

echo "Esecuzione pylint su: $TARGET_FILE"

# Esecuzione di pylint con i filtri condizionali
if [ "$ERRORS_ONLY" = true ] && [ -n "$EXCLUDE_PATTERN" ]; then
    PYTHONPATH=. pylint "$TARGET_FILE" | grep ": E" | grep -vi "$EXCLUDE_PATTERN"
elif [ "$ERRORS_ONLY" = true ]; then
    PYTHONPATH=. pylint "$TARGET_FILE" | grep ": E"
elif [ -n "$EXCLUDE_PATTERN" ]; then
    PYTHONPATH=. pylint "$TARGET_FILE" | grep -vi "$EXCLUDE_PATTERN"
else
    # mettere . o src dipende da come sono impostati
    # gli import nel progetto
    # PYTHONPATH=src pylint "$TARGET_FILE"
    PYTHONPATH=. pylint "$TARGET_FILE"
fi