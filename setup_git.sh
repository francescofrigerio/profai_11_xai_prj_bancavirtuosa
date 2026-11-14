#!/bin/bash

# Interrompe lo script se si verifica un errore
set -e

echo "1. Inizializzazione di Git "
git init

echo "2. Configurazione dell'utente ==="
git config user.name "francescofrigerio"
git config user.email "francesco.frigerio@greencore.it"
git config --list

echo "3. Rinomina del branch principale in 'main' "
git branch -M main

echo "4. Aggiunta di tutti i file e cartelle (inclusi 0_brainstorming e 1_design/fase_0) ==="
git add .

echo "5. Creazione del primo commit locale ==="
git commit -m "Primo commit: struttura iniziale, brainstorming e design"

echo "6. Collegamento al remote repository GitHub "
# Rimuove l'origin se esiste già per evitare errori di "remote origin already exists"
git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/francescofrigerio/profai_11_xai_prj_bancavirtuosa.git

echo "7. Invio del codice su GitHub (push) ==="
git push -u origin main

echo "Configurazione Git completata con successo! ==="
