#!/bin/bash

CSV_FILE="/home/lucasmocellin/tcc/data/docking/predicted_results_with_ids_refined.csv"
ORIGEM="/home/lucasmocellin/tcc/data/docking/inputs/pdbbind-refined-set"
DESTINO="/home/lucasmocellin/tcc/melimda/Data/Complexos"

mkdir -p "$DESTINO"

# Pula o cabeçalho e percorre os IDs
tail -n +2 "$CSV_FILE" | cut -d',' -f1 | while read ID; do
    if [ -d "$ORIGEM/$ID" ]; then
        echo "Copiando $ID..."
        cp -r "$ORIGEM/$ID" "$DESTINO/"
    else
        echo "Pasta $ID não encontrada em $ORIGEM"
    fi
done
