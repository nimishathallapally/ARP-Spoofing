#!/usr/bin/env python3
"""
PCAP to CSV Converter using Scapy (Lightweight Alternative)

This script converts PCAP files to CSV format using scapy library.
It's a simpler, faster alternative to nfstream for basic flow extraction.

Usage:
    python scripts/pcap_to_csv_scapy.py input.pcap output.csv
    python scripts/pcap_to_csv_scapy.py input.pcap output.csv --label normal
"""

import sys
from pathlib import Path
import argparse
import logging
from typing import Optional, Dict, List
import pandas as pd
from collections import defaultdict
from datetime import datetime

# Try to import scapy
try:
    from scapy.all import rdpcap, IP, TCP, UDP, ARP, Ether
    from scapy.layers.inet import ICMP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("⚠ Warning: scapy not installed. Install with: pip install scapy")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from utils import setup_logging, print_header
except:
    def setup_logging(level='INFO'):
        logging.basicConfig(level=level)
    def print_header(text, width=60, char='='):
        print(char * width)
        print(text.center(width))
        print(char * width)

logger = logging.getLogger(__name__)


class FlowExtractor:
    """Extract network flows from PCAP using scapy"""
    
    def __init__(self):
        self.flows = defaultdict(lambda: {
            'packets': [],
            'src_ip': None,
            'dst_ip': None,
            'src_mac': None,
            'dst_mac': None,
            'src_port': 0,
            'dst_port': 0,
            'protocol': 0,
            'first_time': None,
            'last_time': None,
            'bytes': 0,
            'syn_count': 0,
            'ack_count': 0,
            'fin_count': 0,
            'rst_count': 0,
            'psh_count': 0,
        })
    
    def get_flow_key(self, pkt):
        """Generate a unique flow key for bidirectional flow"""
        if IP in pkt:
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            protocol = pkt[IP].proto
            
            # Get ports if TCP/UDP
            if TCP in pkt:
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
            elif UDP in pkt:
                src_port = pkt[UDP].sport
                dst_port = pkt[UDP].dport
            else:
                src_port = 0
                dst_port = 0
            
            # Create bidirectional flow key (sorted to ensure consistency)
            if (src_ip, src_port) < (dst_ip, dst_port):
                return f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
            else:
                return f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}"
        
        return None
    
    def process_packet(self, pkt, pkt_time):
        """Process a single packet and add to flow"""
        flow_key = self.get_flow_key(pkt)
        if not flow_key:
            return
        
        flow = self.flows[flow_key]
        
        # Initialize flow metadata on first packet
        if not flow['packets']:
            if IP in pkt:
                flow['src_ip'] = pkt[IP].src
                flow['dst_ip'] = pkt[IP].dst
                flow['protocol'] = pkt[IP].proto
            
            if Ether in pkt:
                flow['src_mac'] = pkt[Ether].src
                flow['dst_mac'] = pkt[Ether].dst
            
            if TCP in pkt:
                flow['src_port'] = pkt[TCP].sport
                flow['dst_port'] = pkt[TCP].dport
            elif UDP in pkt:
                flow['src_port'] = pkt[UDP].sport
                flow['dst_port'] = pkt[UDP].dport
            
            flow['first_time'] = pkt_time
        
        # Update flow statistics
        flow['last_time'] = pkt_time
        flow['packets'].append(pkt)
        flow['bytes'] += len(pkt)
        
        # Count TCP flags
        if TCP in pkt:
            flags = pkt[TCP].flags
            if flags & 0x02:  # SYN
                flow['syn_count'] += 1
            if flags & 0x10:  # ACK
                flow['ack_count'] += 1
            if flags & 0x01:  # FIN
                flow['fin_count'] += 1
            if flags & 0x04:  # RST
                flow['rst_count'] += 1
            if flags & 0x08:  # PSH
                flow['psh_count'] += 1
    
    def flows_to_dataframe(self, label='unknown'):
        """Convert flows to pandas DataFrame with simplified features"""
        import numpy as np
        records = []
        
        for flow_id, flow in self.flows.items():
            if not flow['packets']:
                continue
            
            # Convert times to float to avoid EDecimal issues
            first_time = float(flow['first_time']) if flow['first_time'] else 0.0
            last_time = float(flow['last_time']) if flow['last_time'] else 0.0
            duration_ms = (last_time - first_time) * 1000 if first_time and last_time else 0.0
            
            packet_count = len(flow['packets'])
            packet_sizes = [len(pkt) for pkt in flow['packets']]
            
            # Calculate packet size statistics - convert to Python native types
            min_ps = int(min(packet_sizes)) if packet_sizes else 0
            max_ps = int(max(packet_sizes)) if packet_sizes else 0
            mean_ps = float(sum(packet_sizes) / len(packet_sizes)) if packet_sizes else 0.0
            # Calculate std manually
            if packet_sizes and len(packet_sizes) > 1:
                mean_val = sum(packet_sizes) / len(packet_sizes)
                variance = sum((x - mean_val) ** 2 for x in packet_sizes) / len(packet_sizes)
                std_ps = float(variance ** 0.5)
            else:
                std_ps = 0.0
            
            # Calculate inter-arrival times - convert to Python native types
            if len(flow['packets']) > 1:
                times = [float(pkt.time) for pkt in flow['packets']]  # Convert EDecimal to float
                iats = [(times[i+1] - times[i]) * 1000 for i in range(len(times)-1)]
                min_iat = float(min(iats)) if iats else 0.0
                max_iat = float(max(iats)) if iats else 0.0
                mean_iat = float(sum(iats) / len(iats)) if iats else 0.0  # Use Python sum/len instead of np.mean
                # Calculate std manually to avoid numpy issues
                if iats and len(iats) > 1:
                    mean_val = sum(iats) / len(iats)
                    variance = sum((x - mean_val) ** 2 for x in iats) / len(iats)
                    std_iat = float(variance ** 0.5)
                else:
                    std_iat = 0.0
            else:
                min_iat = max_iat = mean_iat = std_iat = 0.0
            
            record = {
                # Flow identifiers
                'id': str(flow_id),
                'expiration_id': 0,
                
                # Source/Destination info
                'src_ip': str(flow['src_ip'] or '0.0.0.0'),
                'src_mac': str(flow['src_mac'] or '00:00:00:00:00:00'),
                'src_oui': str(flow['src_mac'][:8] if flow['src_mac'] else '00:00:00'),
                'src_port': int(flow['src_port']),
                'dst_ip': str(flow['dst_ip'] or '0.0.0.0'),
                'dst_mac': str(flow['dst_mac'] or '00:00:00:00:00:00'),
                'dst_oui': str(flow['dst_mac'][:8] if flow['dst_mac'] else '00:00:00'),
                'dst_port': int(flow['dst_port']),
                
                # Protocol info
                'protocol': int(flow['protocol']),
                'ip_version': 4,
                'vlan_id': 0,
                'tunnel_id': 0,
                
                # Temporal features (bidirectional)
                'bidirectional_first_seen_ms': int(first_time * 1000),
                'bidirectional_last_seen_ms': int(last_time * 1000),
                'bidirectional_duration_ms': int(duration_ms),
                'bidirectional_packets': int(packet_count),
                'bidirectional_bytes': int(flow['bytes']),
                
                # Source to destination (simplified - same as bidirectional)
                'src2dst_first_seen_ms': int(first_time * 1000),
                'src2dst_last_seen_ms': int(last_time * 1000),
                'src2dst_duration_ms': int(duration_ms),
                'src2dst_packets': int(packet_count),
                'src2dst_bytes': int(flow['bytes']),
                
                # Destination to source (simplified - zeros for now)
                'dst2src_first_seen_ms': 0,
                'dst2src_last_seen_ms': 0,
                'dst2src_duration_ms': 0,
                'dst2src_packets': 0,
                'dst2src_bytes': 0,
                
                # Packet size statistics (bidirectional)
                'bidirectional_min_ps': min_ps,
                'bidirectional_mean_ps': mean_ps,
                'bidirectional_stddev_ps': std_ps,
                'bidirectional_max_ps': max_ps,
                
                # Source to destination packet size
                'src2dst_min_ps': min_ps,
                'src2dst_mean_ps': mean_ps,
                'src2dst_stddev_ps': std_ps,
                'src2dst_max_ps': max_ps,
                
                # Destination to source packet size (zeros)
                'dst2src_min_ps': 0,
                'dst2src_mean_ps': 0.0,
                'dst2src_stddev_ps': 0.0,
                'dst2src_max_ps': 0,
                
                # Inter-arrival time statistics (bidirectional)
                'bidirectional_min_piat_ms': min_iat,
                'bidirectional_mean_piat_ms': mean_iat,
                'bidirectional_stddev_piat_ms': std_iat,
                'bidirectional_max_piat_ms': max_iat,
                
                # Source to destination IAT
                'src2dst_min_piat_ms': min_iat,
                'src2dst_mean_piat_ms': mean_iat,
                'src2dst_stddev_piat_ms': std_iat,
                'src2dst_max_piat_ms': max_iat,
                
                # Destination to source IAT (zeros)
                'dst2src_min_piat_ms': 0.0,
                'dst2src_mean_piat_ms': 0.0,
                'dst2src_stddev_piat_ms': 0.0,
                'dst2src_max_piat_ms': 0.0,
                
                # TCP flags (bidirectional)
                'bidirectional_syn_packets': int(flow['syn_count']),
                'bidirectional_cwr_packets': 0,
                'bidirectional_ece_packets': 0,
                'bidirectional_urg_packets': 0,
                'bidirectional_ack_packets': int(flow['ack_count']),
                'bidirectional_psh_packets': int(flow['psh_count']),
                'bidirectional_rst_packets': int(flow['rst_count']),
                'bidirectional_fin_packets': int(flow['fin_count']),
                
                # Source to destination TCP flags
                'src2dst_syn_packets': int(flow['syn_count']),
                'src2dst_cwr_packets': 0,
                'src2dst_ece_packets': 0,
                'src2dst_urg_packets': 0,
                'src2dst_ack_packets': int(flow['ack_count']),
                'src2dst_psh_packets': int(flow['psh_count']),
                'src2dst_rst_packets': int(flow['rst_count']),
                'src2dst_fin_packets': int(flow['fin_count']),
                
                # Destination to source TCP flags (zeros)
                'dst2src_syn_packets': 0,
                'dst2src_cwr_packets': 0,
                'dst2src_ece_packets': 0,
                'dst2src_urg_packets': 0,
                'dst2src_ack_packets': 0,
                'dst2src_psh_packets': 0,
                'dst2src_rst_packets': 0,
                'dst2src_fin_packets': 0,
                
                # Application layer (simplified)
                'application_name': '',
                'application_category_name': '',
                'application_is_guessed': 0,
                'application_confidence': 0,
                'requested_server_name': '',
                'client_fingerprint': '',
                'server_fingerprint': '',
                'user_agent': '',
                'content_type': '',
                
                # Label
                'Label': label
            }
            
            records.append(record)
        
        return pd.DataFrame(records)


def convert_pcap_to_csv_scapy(
    pcap_file: str,
    output_file: str,
    label: str = 'unknown',
    max_packets: Optional[int] = None
) -> bool:
    """
    Convert PCAP to CSV using scapy.
    
    Args:
        pcap_file: Path to PCAP file
        output_file: Path to output CSV
        label: Label for flows
        max_packets: Maximum packets to read
    
    Returns:
        True if successful
    """
    if not SCAPY_AVAILABLE:
        logger.error("scapy library not installed")
        logger.error("Install with: pip install scapy")
        return False
    
    pcap_path = Path(pcap_file)
    if not pcap_path.exists():
        logger.error(f"PCAP file not found: {pcap_file}")
        return False
    
    try:
        logger.info(f"Reading PCAP file: {pcap_file}")
        
        # Read PCAP file
        packets = rdpcap(str(pcap_path))
        total_packets = len(packets)
        
        logger.info(f"Total packets: {total_packets:,}")
        
        # Extract flows
        extractor = FlowExtractor()
        
        packets_to_process = packets[:max_packets] if max_packets else packets
        logger.info(f"Processing {len(packets_to_process):,} packets...")
        
        for i, pkt in enumerate(packets_to_process):
            if i % 1000 == 0 and i > 0:
                logger.info(f"  Processed {i:,} packets...")
            
            extractor.process_packet(pkt, float(pkt.time))
        
        logger.info(f"✓ Extracted {len(extractor.flows)} flows")
        
        # Convert to DataFrame
        df = extractor.flows_to_dataframe(label=label)
        
        logger.info(f"✓ Created DataFrame: {len(df)} rows, {len(df.columns)} columns")
        
        # Save to CSV
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        
        logger.info(f"✓ Saved to: {output_file}")
        
        # Display statistics
        print_header("CONVERSION COMPLETE", width=60, char='-')
        print(f"PCAP file: {pcap_file}")
        print(f"CSV file: {output_file}")
        print(f"Total packets: {total_packets:,}")
        print(f"Total flows: {len(df):,}")
        print(f"Total bytes: {df['bidirectional_bytes'].sum():,}")
        print(f"Label: {label}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error converting PCAP: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Convert PCAP to CSV using Scapy (Lightweight)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('pcap_file', help='Input PCAP file')
    parser.add_argument('output_file', help='Output CSV file')
    parser.add_argument('--label', default='unknown', help='Label for flows')
    parser.add_argument('--max-packets', type=int, help='Max packets to process')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(level='DEBUG' if args.verbose else 'INFO')
    
    print_header("PCAP TO CSV CONVERTER (SCAPY)", width=60)
    
    if not SCAPY_AVAILABLE:
        print("\n❌ Error: scapy not installed")
        print("Install with: pip install scapy")
        return 1
    
    success = convert_pcap_to_csv_scapy(
        pcap_file=args.pcap_file,
        output_file=args.output_file,
        label=args.label,
        max_packets=args.max_packets
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
