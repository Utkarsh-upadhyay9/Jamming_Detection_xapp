#!/usr/bin/env python3
"""
Real USRP Data Analyzer and Synthetic Dataset Generator
Analyzes real srsRAN gNodeB metrics and generates realistic jamming detection dataset
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple
import json
from datetime import datetime, timedelta

class USRPDataAnalyzer:
    def __init__(self, metrics_file: str):
        self.metrics_file = metrics_file
        self.parsed_data = []
        self.normal_patterns = {}
        self.jamming_indicators = {}
        
    def parse_usrp_metrics(self) -> List[Dict]:
        """Parse real USRP metrics from log file"""
        print("📊 Analyzing real USRP metrics...")
        
        with open(self.metrics_file, 'r') as f:
            content = f.read()
        
        # Pattern to extract metrics lines
        pattern = r'\s*(\d+)\s+([0-9a-fA-F]+)\s+\|\s*(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+)\s+([0-9.kM]+|0)\s+(\d+)\s+(\d+)\s+(\d+)%\s+([0-9.kM]+|0)\s+\|\s*([\d.-]+|n/a)\s*([\d.-]+|n/a)\s+(\d+)\s+(\d+)\s+([0-9.kM]+|0)\s+(\d+)\s+(\d+)\s+(\d+)%\s+([0-9.kM]+|0)\s+([-\d]+n|[-\d]+)\s+([-\d]+|n/a)'
        
        matches = re.findall(pattern, content)
        
        parsed_metrics = []
        for i, match in enumerate(matches):
            try:
                # Parse DL metrics
                pci = int(match[0])
                rnti = match[1]
                dl_cqi = float(match[2])
                dl_ri = float(match[3])
                dl_mcs = int(match[4])
                dl_brate = self._parse_rate(match[5])
                dl_ok = int(match[6])
                dl_nok = int(match[7])
                dl_err_pct = int(match[8])
                dl_bs = self._parse_size(match[9])
                
                # Parse UL metrics
                ul_pusch = float(match[10]) if match[10] != 'n/a' else np.nan
                ul_rsrp = float(match[11]) if match[11] != 'n/a' else np.nan
                ul_ri = int(match[12])
                ul_mcs = int(match[13])
                ul_brate = self._parse_rate(match[14])
                ul_ok = int(match[15])
                ul_nok = int(match[16])
                ul_err_pct = int(match[17])
                ul_bsr = self._parse_size(match[18])
                ul_ta = self._parse_timing(match[19])
                ul_phr = float(match[20]) if match[20] != 'n/a' else np.nan
                
                metrics = {
                    'timestamp': i,  # Sequence number as timestamp
                    'pci': pci,
                    'rnti': rnti,
                    'dl_cqi': dl_cqi,
                    'dl_ri': dl_ri,
                    'dl_mcs': dl_mcs,
                    'dl_brate': dl_brate,
                    'dl_throughput_mbps': dl_brate / 1_000_000,
                    'dl_ok': dl_ok,
                    'dl_nok': dl_nok,
                    'dl_error_rate': dl_err_pct / 100.0,
                    'dl_buffer_size': dl_bs,
                    'ul_pusch_snr': ul_pusch,
                    'ul_rsrp': ul_rsrp,
                    'ul_ri': ul_ri,
                    'ul_mcs': ul_mcs,
                    'ul_brate': ul_brate,
                    'ul_throughput_mbps': ul_brate / 1_000_000,
                    'ul_ok': ul_ok,
                    'ul_nok': ul_nok,
                    'ul_error_rate': ul_err_pct / 100.0,
                    'ul_buffer_status': ul_bsr,
                    'timing_advance': ul_ta,
                    'power_headroom': ul_phr
                }
                
                parsed_metrics.append(metrics)
                
            except (ValueError, IndexError) as e:
                print(f"⚠️  Skipping malformed line {i}: {e}")
                continue
        
        print(f"✅ Parsed {len(parsed_metrics)} valid metrics samples")
        return parsed_metrics
    
    def _parse_rate(self, rate_str: str) -> float:
        """Parse data rate string (e.g., '1.2M', '500k') to bps"""
        if rate_str == '0':
            return 0.0
        
        rate_str = rate_str.lower()
        if 'm' in rate_str:
            return float(rate_str.replace('m', '')) * 1_000_000
        elif 'k' in rate_str:
            return float(rate_str.replace('k', '')) * 1_000
        else:
            return float(rate_str)
    
    def _parse_size(self, size_str: str) -> float:
        """Parse buffer size string"""
        if size_str == '0':
            return 0.0
        
        size_str = size_str.lower()
        if 'm' in size_str:
            return float(size_str.replace('m', '')) * 1_000_000
        elif 'k' in size_str:
            return float(size_str.replace('k', '')) * 1_000
        else:
            return float(size_str)
    
    def _parse_timing(self, timing_str: str) -> float:
        """Parse timing advance (e.g., '233n', '-285n')"""
        if timing_str == 'n/a':
            return np.nan
        
        # Remove 'n' suffix and convert to nanoseconds
        return float(timing_str.replace('n', ''))
    
    def analyze_patterns(self, metrics: List[Dict]) -> Dict:
        """Analyze normal vs jamming patterns"""
        print("🔍 Analyzing traffic patterns...")
        
        df = pd.DataFrame(metrics)
        
        # Calculate derived features
        df['total_throughput'] = df['dl_throughput_mbps'] + df['ul_throughput_mbps']
        df['combined_error_rate'] = (df['dl_error_rate'] + df['ul_error_rate']) / 2
        df['snr_rsrp_diff'] = df['ul_pusch_snr'] - df['ul_rsrp']
        
        # Identify potential jamming incidents (high error rates, low RSRP, etc.)
        jamming_threshold = {
            'high_error_rate': df['combined_error_rate'].quantile(0.95),
            'low_rsrp': df['ul_rsrp'].quantile(0.05),
            'low_throughput': df['total_throughput'].quantile(0.10)
        }
        
        # Mark potential jamming samples
        df['is_suspicious'] = (
            (df['combined_error_rate'] > jamming_threshold['high_error_rate']) |
            (df['ul_rsrp'] < jamming_threshold['low_rsrp']) |
            (df['total_throughput'] < jamming_threshold['low_throughput'])
        )
        
        print(f"📈 Found {df['is_suspicious'].sum()} suspicious samples out of {len(df)}")
        
        # Normal traffic statistics
        normal_data = df[~df['is_suspicious']]
        patterns = {
            'normal': {
                'dl_cqi': {'mean': normal_data['dl_cqi'].mean(), 'std': normal_data['dl_cqi'].std()},
                'dl_mcs': {'mean': normal_data['dl_mcs'].mean(), 'std': normal_data['dl_mcs'].std()},
                'ul_rsrp': {'mean': normal_data['ul_rsrp'].mean(), 'std': normal_data['ul_rsrp'].std()},
                'ul_pusch_snr': {'mean': normal_data['ul_pusch_snr'].mean(), 'std': normal_data['ul_pusch_snr'].std()},
                'dl_throughput': {'mean': normal_data['dl_throughput_mbps'].mean(), 'std': normal_data['dl_throughput_mbps'].std()},
                'ul_throughput': {'mean': normal_data['ul_throughput_mbps'].mean(), 'std': normal_data['ul_throughput_mbps'].std()},
                'dl_error_rate': {'mean': normal_data['dl_error_rate'].mean(), 'std': normal_data['dl_error_rate'].std()},
                'ul_error_rate': {'mean': normal_data['ul_error_rate'].mean(), 'std': normal_data['ul_error_rate'].std()},
                'timing_advance': {'mean': normal_data['timing_advance'].mean(), 'std': normal_data['timing_advance'].std()},
                'power_headroom': {'mean': normal_data['power_headroom'].mean(), 'std': normal_data['power_headroom'].std()}
            },
            'suspicious': {
                'count': int(df['is_suspicious'].sum()),
                'percentage': float(df['is_suspicious'].sum() / len(df) * 100)
            }
        }
        
        return patterns

def generate_realistic_dataset(patterns: Dict, target_samples: int = 25000) -> pd.DataFrame:
    """Generate realistic jamming detection dataset based on USRP patterns"""
    print(f"🎯 Generating realistic dataset with {target_samples} samples...")
    
    # Target distribution
    normal_samples = int(target_samples * 0.70)  # 70%
    power_jamming_samples = int(target_samples * 0.10)  # 10%
    sweep_jamming_samples = int(target_samples * 0.10)  # 10%
    intelligent_jamming_samples = int(target_samples * 0.10)  # 10%
    
    all_samples = []
    
    # Generate normal traffic (based on real patterns)
    print("📡 Generating normal traffic samples...")
    normal_stats = patterns['normal']
    
    for i in range(normal_samples):
        timestamp = datetime.now() + timedelta(seconds=i*0.1)
        
        # Base metrics from real data patterns
        dl_cqi = np.clip(np.random.normal(normal_stats['dl_cqi']['mean'], normal_stats['dl_cqi']['std']), 1, 15)
        dl_mcs = np.clip(np.random.normal(normal_stats['dl_mcs']['mean'], normal_stats['dl_mcs']['std']), 0, 28)
        ul_rsrp = np.clip(np.random.normal(normal_stats['ul_rsrp']['mean'], normal_stats['ul_rsrp']['std']), -50, -10)
        ul_pusch_snr = np.clip(np.random.normal(normal_stats['ul_pusch_snr']['mean'], normal_stats['ul_pusch_snr']['std']), 10, 40)
        
        # Throughput with realistic variations
        dl_throughput = np.clip(np.random.normal(normal_stats['dl_throughput']['mean'], normal_stats['dl_throughput']['std']), 0, 100)
        ul_throughput = np.clip(np.random.normal(normal_stats['ul_throughput']['mean'], normal_stats['ul_throughput']['std']), 0, 50)
        
        # Low error rates for normal traffic
        dl_error_rate = np.clip(np.random.exponential(0.005), 0, 0.05)
        ul_error_rate = np.clip(np.random.exponential(0.005), 0, 0.05)
        
        # Timing and power metrics
        timing_advance = np.random.normal(normal_stats['timing_advance']['mean'], normal_stats['timing_advance']['std'])
        power_headroom = np.clip(np.random.normal(normal_stats['power_headroom']['mean'], normal_stats['power_headroom']['std']), 0, 30)
        
        # Calculate the 15 features used in the ensemble model
        sample = create_feature_vector(
            timestamp, dl_cqi, dl_mcs, ul_rsrp, ul_pusch_snr,
            dl_throughput, ul_throughput, dl_error_rate, ul_error_rate,
            timing_advance, power_headroom, attack_type='normal'
        )
        
        all_samples.append(sample)
    
    # Generate power jamming samples
    print("⚡ Generating power jamming samples...")
    for i in range(power_jamming_samples):
        timestamp = datetime.now() + timedelta(seconds=(normal_samples + i)*0.1)
        
        # Power jamming characteristics: severely degraded RSRP and SNR
        dl_cqi = np.clip(np.random.normal(8, 3), 1, 15)  # Reduced CQI
        dl_mcs = np.clip(np.random.normal(10, 5), 0, 28)  # Lower MCS
        ul_rsrp = np.clip(np.random.normal(-35, 5), -50, -25)  # Severely degraded RSRP
        ul_pusch_snr = np.clip(np.random.normal(15, 3), 5, 25)  # Reduced SNR
        
        # Significantly reduced throughput
        dl_throughput = np.clip(np.random.normal(5, 2), 0, 15)
        ul_throughput = np.clip(np.random.normal(1, 0.5), 0, 3)
        
        # High error rates
        dl_error_rate = np.clip(np.random.normal(0.15, 0.05), 0.05, 0.40)
        ul_error_rate = np.clip(np.random.normal(0.20, 0.05), 0.10, 0.50)
        
        # Affected timing and power
        timing_advance = np.random.normal(250, 50)
        power_headroom = np.clip(np.random.normal(5, 2), 0, 10)
        
        sample = create_feature_vector(
            timestamp, dl_cqi, dl_mcs, ul_rsrp, ul_pusch_snr,
            dl_throughput, ul_throughput, dl_error_rate, ul_error_rate,
            timing_advance, power_headroom, attack_type='power_jamming'
        )
        
        all_samples.append(sample)
    
    # Generate sweep jamming samples  
    print("🌊 Generating sweep jamming samples...")
    for i in range(sweep_jamming_samples):
        timestamp = datetime.now() + timedelta(seconds=(normal_samples + power_jamming_samples + i)*0.1)
        
        # Sweep jamming: periodic interference patterns
        sweep_phase = (i % 20) / 20.0  # 20-sample sweep cycle
        interference_intensity = 0.5 + 0.5 * np.sin(2 * np.pi * sweep_phase)
        
        # Variable degradation based on sweep phase
        dl_cqi = np.clip(np.random.normal(12 - 4*interference_intensity, 2), 1, 15)
        dl_mcs = np.clip(np.random.normal(20 - 8*interference_intensity, 3), 0, 28)
        ul_rsrp = np.clip(np.random.normal(-25 - 10*interference_intensity, 3), -45, -15)
        ul_pusch_snr = np.clip(np.random.normal(25 - 8*interference_intensity, 2), 10, 35)
        
        # Variable throughput
        dl_throughput = np.clip(np.random.normal(20 - 15*interference_intensity, 3), 0, 30)
        ul_throughput = np.clip(np.random.normal(8 - 6*interference_intensity, 2), 0, 12)
        
        # Variable error rates
        dl_error_rate = np.clip(np.random.normal(0.05 + 0.15*interference_intensity, 0.02), 0, 0.30)
        ul_error_rate = np.clip(np.random.normal(0.08 + 0.20*interference_intensity, 0.03), 0, 0.40)
        
        timing_advance = np.random.normal(220 + 30*interference_intensity, 20)
        power_headroom = np.clip(np.random.normal(12 - 6*interference_intensity, 2), 0, 20)
        
        sample = create_feature_vector(
            timestamp, dl_cqi, dl_mcs, ul_rsrp, ul_pusch_snr,
            dl_throughput, ul_throughput, dl_error_rate, ul_error_rate,
            timing_advance, power_headroom, attack_type='sweep_jamming'
        )
        
        all_samples.append(sample)
    
    # Generate intelligent jamming samples
    print("🧠 Generating intelligent jamming samples...")
    for i in range(intelligent_jamming_samples):
        timestamp = datetime.now() + timedelta(seconds=(normal_samples + power_jamming_samples + sweep_jamming_samples + i)*0.1)
        
        # Intelligent jamming: adaptive, targets specific channels/users
        # More subtle than power jamming but targeted
        
        dl_cqi = np.clip(np.random.normal(10, 2), 1, 15)
        dl_mcs = np.clip(np.random.normal(15, 4), 0, 28)
        ul_rsrp = np.clip(np.random.normal(-28, 3), -40, -20)
        ul_pusch_snr = np.clip(np.random.normal(20, 3), 10, 30)
        
        # Moderate throughput reduction
        dl_throughput = np.clip(np.random.normal(12, 4), 0, 25)
        ul_throughput = np.clip(np.random.normal(4, 2), 0, 8)
        
        # Moderate but consistent error rates
        dl_error_rate = np.clip(np.random.normal(0.08, 0.03), 0.02, 0.20)
        ul_error_rate = np.clip(np.random.normal(0.12, 0.04), 0.05, 0.25)
        
        timing_advance = np.random.normal(240, 25)
        power_headroom = np.clip(np.random.normal(8, 3), 0, 15)
        
        sample = create_feature_vector(
            timestamp, dl_cqi, dl_mcs, ul_rsrp, ul_pusch_snr,
            dl_throughput, ul_throughput, dl_error_rate, ul_error_rate,
            timing_advance, power_headroom, attack_type='intelligent_jamming'
        )
        
        all_samples.append(sample)
    
    # Create DataFrame
    df = pd.DataFrame(all_samples)
    
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    print(f"✅ Generated {len(df)} total samples:")
    print(f"   📊 Normal traffic: {(df['attack_type'] == 'normal').sum()}")
    print(f"   ⚡ Power jamming: {(df['attack_type'] == 'power_jamming').sum()}")
    print(f"   🌊 Sweep jamming: {(df['attack_type'] == 'sweep_jamming').sum()}")
    print(f"   🧠 Intelligent jamming: {(df['attack_type'] == 'intelligent_jamming').sum()}")
    
    return df

def create_feature_vector(timestamp, dl_cqi, dl_mcs, ul_rsrp, ul_pusch_snr,
                         dl_throughput, ul_throughput, dl_error_rate, ul_error_rate,
                         timing_advance, power_headroom, attack_type):
    """Create the 15-feature vector used in the ensemble model"""
    
    # Add some realistic noise and correlations
    total_throughput = dl_throughput + ul_throughput
    avg_error_rate = (dl_error_rate + ul_error_rate) / 2
    snr_rsrp_ratio = ul_pusch_snr / abs(ul_rsrp) if ul_rsrp != 0 else 0
    
    # Calculate packet loss indicators
    dl_packets_total = max(1, int(dl_throughput * 100))  # Approximate packets
    ul_packets_total = max(1, int(ul_throughput * 100))
    dl_packets_lost = int(dl_packets_total * dl_error_rate)
    ul_packets_lost = int(ul_packets_total * ul_error_rate)
    
    # Channel quality variations
    cqi_variation = np.random.normal(0, 0.5)
    mcs_efficiency = dl_mcs / 28.0  # Normalized MCS
    
    return {
        'timestamp': timestamp,
        'throughput_mbps': total_throughput,
        'packet_loss_rate': avg_error_rate,
        'rsrp_dbm': ul_rsrp,
        'snr_db': ul_pusch_snr,
        'cqi': dl_cqi + cqi_variation,
        'mcs_dl': dl_mcs,
        'mcs_ul': max(0, dl_mcs - np.random.randint(0, 3)),  # UL typically lower
        'timing_advance_ns': timing_advance,
        'power_headroom_db': power_headroom,
        'dl_error_rate': dl_error_rate,
        'ul_error_rate': ul_error_rate,
        'snr_rsrp_ratio': snr_rsrp_ratio,
        'mcs_efficiency': mcs_efficiency,
        'channel_utilization': min(1.0, total_throughput / 100.0),  # Normalized
        'attack_type': attack_type
    }

def main():
    print("🚀 Real USRP Data Analysis and Synthetic Dataset Generation")
    print("=" * 60)
    
    # Analyze real USRP data
    analyzer = USRPDataAnalyzer('real_metrics_USRP.txt')
    real_metrics = analyzer.parse_usrp_metrics()
    
    if not real_metrics:
        print("❌ No valid metrics found in the file")
        return
    
    patterns = analyzer.analyze_patterns(real_metrics)
    
    # Save patterns analysis
    with open('real_usrp_patterns.json', 'w') as f:
        # Convert numpy types to regular Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Deep convert the patterns
        json_patterns = json.loads(json.dumps(patterns, default=convert_numpy))
        json.dump(json_patterns, f, indent=2)
    
    print("💾 Saved real USRP patterns to: real_usrp_patterns.json")
    
    # Generate synthetic dataset
    synthetic_df = generate_realistic_dataset(patterns, target_samples=25000)
    
    # Split into normal and jamming datasets for compatibility
    normal_data = synthetic_df[synthetic_df['attack_type'] == 'normal'].copy()
    jamming_data = synthetic_df[synthetic_df['attack_type'] != 'normal'].copy()
    
    # Save datasets
    normal_data.to_csv('realistic_normal_traffic.csv', index=False)
    jamming_data.to_csv('realistic_jamming_attacks.csv', index=False)
    
    print("✅ Realistic datasets saved:")
    print(f"   📄 realistic_normal_traffic.csv: {len(normal_data)} samples")
    print(f"   📄 realistic_jamming_attacks.csv: {len(jamming_data)} samples")
    
    # Generate summary statistics
    print("\n📈 Dataset Statistics:")
    print("-" * 40)
    for attack_type in synthetic_df['attack_type'].unique():
        subset = synthetic_df[synthetic_df['attack_type'] == attack_type]
        print(f"\n{attack_type.upper()}:")
        print(f"  Samples: {len(subset)}")
        print(f"  Avg Throughput: {subset['throughput_mbps'].mean():.2f} Mbps")
        print(f"  Avg RSRP: {subset['rsrp_dbm'].mean():.2f} dBm")
        print(f"  Avg Error Rate: {subset['packet_loss_rate'].mean():.3f}")
        print(f"  Avg SNR: {subset['snr_db'].mean():.2f} dB")

if __name__ == "__main__":
    main()
