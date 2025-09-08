# Ensemble_ML_Jamming_detection_dataset

## Overview
This dataset contains 25,000 network traffic samples for jamming detection in Open Radio Access Networks (O-RAN), collected using MAC layer metrics through the E2 interface.

## Dataset Composition
- **Total Samples**: 25,000
- **Normal Traffic**: 17,500 samples (70%)
- **Jamming Attacks**: 7,500 samples (30%)
  - Power Jamming: 2,500 samples
  - Sweep Jamming: 2,500 samples  
  - Intelligent Jamming: 2,500 samples


## Features (15 total)
### Signal Quality Indicators
- `sinr_mean`: Mean Signal-to-Interference-plus-Noise Ratio
- `sinr_std`: SINR standard deviation
- `rsrp_mean`: Mean Reference Signal Received Power
- `rsrp_std`: RSRP standard deviation
- `rsrq_mean`: Mean Reference Signal Received Quality

### Traffic Characteristics
- `ul_throughput`: Uplink throughput (bps)
- `dl_throughput`: Downlink throughput (bps)
- `retx_rate`: Retransmission rate
- `packet_loss_rate`: Packet loss rate

### Temporal Dynamics
- `inter_arrival_variance`: Inter-packet arrival time variance
- `buffer_occupancy`: Buffer occupancy ratio

### Resource Utilization
- `prb_utilization_ul`: Uplink Physical Resource Block utilization
- `prb_utilization_dl`: Downlink Physical Resource Block utilization
- `grant_count`: Scheduling grant count
- `cqi_variance`: Channel Quality Indicator variance

## Network Environments
- **Ideal**: Minimal interference (20% of samples)
- **Moderate**: Controlled interference (30% of samples)
- **Realistic**: High interference with multipath (50% of samples)

## Usage

### Python
```python
import pandas as pd
# Load datasets
normal_data = pd.read_csv('dataset/normal_traffic.csv')
jamming_data = pd.read_csv('dataset/jamming_attacks.csv')
