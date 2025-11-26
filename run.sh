#!/bin/bash

CSV_FILE="data.csv"

if [ ! -f "$CSV_FILE" ]; then
    echo "question,temporality,feature,label,mode,model,f1_train,f1_test" > "$CSV_FILE"
fi

# Listas de valores para iterar
questions=(1 2 3 4 5)
features="All features"
label=("depression" "anxiety")
temporality="False"
mode=("selection feature" "PCA" "t-SNE" "SMOTE" "Pass")

for q in "${questions[@]}"; do
    for l in "${label[@]}"; do
        for m in "${mode[@]}"; do
            python main.py --question "$q" --feature "$f" --label "$l" --temporality "$temporality" --mode "$m" >> "$CSV_FILE"
        done
    done
done