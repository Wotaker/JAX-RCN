#!/bin/bash

OUTPUT_FILE="all_results.csv"

cd "numeric"
echo "Framework,Time,N_Samples,Hidden_Nodes" > "$OUTPUT_FILE"

for file in *.csv; do
    if [[ "$file" != "$OUTPUT_FILE" ]]; then
        cat "$file" >> "$OUTPUT_FILE"
    fi
done

mv "$OUTPUT_FILE" "../$OUTPUT_FILE"
