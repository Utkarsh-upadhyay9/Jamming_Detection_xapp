#!/bin/bash
# Monitor experiment progress

echo "==================================="
echo "EXPERIMENT PROGRESS MONITOR"
echo "==================================="

while true; do
    clear
    echo "==================================="
    echo "EXPERIMENT PROGRESS MONITOR"
    echo "$(date)"
    echo "==================================="
    echo ""
    
    # Check if results exist
    if [ -f "results/traffic_improved_log.txt" ]; then
        echo "📊 TRAFFIC FLOW EXPERIMENT:"
        tail -n 20 results/traffic_improved_log.txt 2>/dev/null || echo "  Running..."
    fi
    
    if [ -f "results/traffic_flow_improved_results.json" ]; then
        echo ""
        echo "✅ Traffic Flow COMPLETE!"
        python3 -c "import json; d=json.load(open('results/traffic_flow_improved_results.json')); print(f'  F1-Score: {d[\"weighted_f1\"]:.4f}'); print(f'  CV F1: {d[\"cv_mean\"]:.4f} ± {d[\"cv_std\"]:.4f}')"
    fi
    
    if [ -f "results/mobility_improved_results.json" ]; then
        echo ""
        echo "✅ Mobility COMPLETE!"
        python3 -c "import json; d=json.load(open('results/mobility_improved_results.json')); print(f'  High: {d[\"High Mobility\"][\"weighted_f1\"]:.4f}'); print(f'  Low: {d[\"Low Mobility\"][\"weighted_f1\"]:.4f}')"
    fi
    
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    sleep 5
done
