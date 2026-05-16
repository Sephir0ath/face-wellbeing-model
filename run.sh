#!/bin/bash

CSV_FILE="data_2.csv"

if [ ! -f "$CSV_FILE" ]; then
    echo "question,temporality,feature,label,mode,model,f1_binary_mean,f1_binary_std,f1_macro_mean,f1_macro_std" > "$CSV_FILE"
fi

# Listas de valores para iterar
questions=(1 2 3 4 5)
features="All features"
label=("depression" "anxiety")
temporality="False"
mode=("selection feature" "PCA" "SMOTE" "Pass") # t-SNE no se usa para entrenamiento, solo visualización


for q in "${questions[@]}"; do
    for l in "${label[@]}"; do
        for m in "${mode[@]}"; do
            python main.py --question "$q" --feature "$features" --label "$l" --temporality "$temporality" --mode "$m" --tune --n_iter 25 --inner_splits 3 >> "$CSV_FILE"
        done
    done
done
