#!/bin/bash
#
if [ $# -lt 2 ]; then
    echo "Usage: $0 <csv_filter_regex_pattern> <log_file> <timezone_shift>"
    echo "The script skips those ending with _label.csv"
    echo "Example: ./apply_mark_anomaly.sh \"^merge_ob_2024-03-05.*\.csv$\" \"./what_the.log\" 8"
    exit 1
fi

pattern="$1"
log_file="$2"
timezone_shift="$3"

for file in *.csv; do
    if [[ $file =~ $pattern && ! $file =~ _label\.csv$ ]]; then
        echo "Processing $file with log $log_file..."
        python3 ./mark_anomaly.py "$file" "$log_file" "$timezone_shift"
    fi
done

echo "Processing complete."
