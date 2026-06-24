#!/bin/bash
set -e

# Move to script directory
cd "$(dirname "$0")"

# Directories
CONFIG_DIR="configs/Generic"
RESULTS_DIR="results"
MODELS_FILE="../models.txt"

echo "=== Extracting datasets ==="

cd "$CONFIG_DIR"

# Extract datasets
for zipfile in data_*.zip; do
    [ -f "$zipfile" ] || continue

    folder_name="${zipfile%.zip}"

    if [ ! -d "$folder_name" ]; then
        echo "Extracting $zipfile..."
        unzip -o -q "$zipfile"
    else
        echo "Folder $folder_name already exists, skipping."
    fi
done

# Dataset list
datasets=()

for dir in data_*/; do
    [ -d "$dir" ] || continue
    datasets+=("${dir%/}")
done

echo "Found ${#datasets[@]} datasets: ${datasets[*]}"

# Check for OSM PBF files in datasets
echo ""
echo "=== Checking for Valhalla map files ==="
for dataset in "${datasets[@]}"; do
    pbf_count=$(find "$dataset" -name "*.osm.pbf" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$pbf_count" -gt 0 ]; then
        echo "  $dataset: $pbf_count .osm.pbf file(s) found"
    else
        echo "  $dataset: no .osm.pbf files"
    fi
done

cd ../..

# Check pyvalhalla availability
if python3 -c "import valhalla" 2>/dev/null; then
    echo "pyvalhalla: available"
else
    echo "pyvalhalla: NOT installed (Valhalla in-process routing disabled)"
    echo "  Install with: pip install pyvalhalla"
fi

# Read models
models=()

while IFS= read -r line || [ -n "$line" ]; do
    [ -n "$line" ] && models+=("$line")
done < "$MODELS_FILE"

echo "Found ${#models[@]} models: ${models[*]}"

# Results dir
mkdir -p "$RESULTS_DIR"

# Backup config
cp "$CONFIG_DIR/config.json" "$CONFIG_DIR/config.json.orig"

# Process models
for model in "${models[@]}"; do

    echo ""
    echo "========================================="
    echo "Processing model: $model"
    echo "========================================="

    # Update model in config.json
    sed -i \
        "s|\"chat_model\": \".*\"|\"chat_model\": \"ollama/$model\"|" \
        "$CONFIG_DIR/config.json"

    echo "Updated config.json with model: ollama/$model"

    model_results_dir="$RESULTS_DIR/$model"
    mkdir -p "$model_results_dir"

    # Process datasets
    for dataset in "${datasets[@]}"; do

        echo ""
        echo "  --- Processing dataset: $dataset ---"

        survey_file="$CONFIG_DIR/$dataset/survey.csv"
        dict_file="$CONFIG_DIR/$dataset/dictionary.csv"

        # Verify files exist
        if [ ! -f "$survey_file" ]; then
            echo "    ERROR: Missing survey file:"
            echo "    $survey_file"
            continue
        fi

        if [ ! -f "$dict_file" ]; then
            echo "    ERROR: Missing dictionary file:"
            echo "    $dict_file"
            continue
        fi

        # Create run folder
        timestamp=$(date +%Y%m%d_%H%M%S)

        run_folder="run/${model}_${dataset}_${timestamp}"

        mkdir -p "$run_folder"

        # Run main.py
        echo "    Running main.py..."

        python3 main.py \
            "$survey_file" \
            "$dict_file" \
            2>&1 | tee "$model_results_dir/${dataset}_run_output.log"

        # Save outputs
        dataset_results_dir="$model_results_dir/$dataset"

        mkdir -p "$dataset_results_dir"

        echo "    Copying results..."

        cp -r "$run_folder"/* \
            "$dataset_results_dir/" 2>/dev/null || true

        # Save oracle/config
        mkdir -p "$dataset_results_dir/oracle"

        cp "$CONFIG_DIR/sampled_oracle.csv" \
            "$dataset_results_dir/oracle/sampled_oracle.csv"

        cp "$CONFIG_DIR/config.json" \
            "$dataset_results_dir/oracle/config.json"

        echo "    Completed: $dataset"
    done

    echo ""
    echo "Completed model: $model"

done

# Restore original config
mv "$CONFIG_DIR/config.json.orig" "$CONFIG_DIR/config.json"

echo ""
echo "========================================="
echo "All done!"
echo "Results saved to: $RESULTS_DIR"
echo "========================================="