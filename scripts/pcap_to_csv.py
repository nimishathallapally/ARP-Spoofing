#!/usr/bin/env python3
"""
PCAP to CSV Converter for ARP Spoofing Detection

This script converts PCAP files to CSV format with the same features
used in training datasets. It extracts network flow features using
nfstream library, which provides bidirectional flow analysis.

Features extracted:
- Flow metadata (IPs, MACs, ports, protocols)
- Temporal features (duration, timestamps)
- Statistical features (packet/byte counts, rates)
- Packet size statistics (min, mean, stddev, max)
- Inter-arrival time statistics
- TCP flags
- Application layer information

Usage:
    python scripts/pcap_to_csv.py input.pcap output.csv
    python scripts/pcap_to_csv.py input.pcap output.csv --label normal
    python scripts/pcap_to_csv.py input.pcap output.csv --label arp_spoofing
"""

import sys
from pathlib import Path
import argparse
import logging
from typing import Optional
import pandas as pd
from tqdm import tqdm

# Try to import nfstream
try:
    from nfstream import NFStreamer
    NFSTREAM_AVAILABLE = True
except ImportError:
    NFSTREAM_AVAILABLE = False
    print("⚠ Warning: nfstream not installed. Install with: pip install nfstream")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils import setup_logging, print_header

logger = logging.getLogger(__name__)


def convert_pcap_to_csv(
    pcap_file: str,
    output_file: str,
    label: Optional[str] = None,
    max_flows: Optional[int] = None
) -> bool:
    """
    Convert PCAP file to CSV with network flow features.
    
    Args:
        pcap_file: Path to input PCAP file
        output_file: Path to output CSV file
        label: Optional label for flows ('normal', 'arp_spoofing', etc.)
        max_flows: Maximum number of flows to process (None = all)
        
    Returns:
        True if successful, False otherwise
    """
    if not NFSTREAM_AVAILABLE:
        logger.error("nfstream library is not installed")
        logger.error("Install it with: pip install nfstream")
        return False
    
    pcap_path = Path(pcap_file)
    if not pcap_path.exists():
        logger.error(f"PCAP file not found: {pcap_file}")
        return False
    
    logger.info(f"Processing PCAP file: {pcap_file}")
    logger.info(f"Output CSV file: {output_file}")
    if label:
        logger.info(f"Label: {label}")
    
    try:
        # Initialize NFStreamer
        # Parameters configured to match training dataset features
        streamer = NFStreamer(
            source=str(pcap_path),
            decode_tunnels=True,
            bpf_filter=None,
            promiscuous_mode=True,
            snapshot_length=1536,
            idle_timeout=120,
            active_timeout=1800,
            accounting_mode=0,  # Standard mode
            udps=None,
            n_dissections=20,
            statistical_analysis=True,
            splt_analysis=0,
            n_meters=0,
            performance_report=0
        )
        
        flows = []
        flow_count = 0
        
        logger.info("Extracting flows from PCAP...")
        
        # Process flows with progress bar
        for flow in tqdm(streamer, desc="Processing flows", unit="flow"):
            # Extract flow features matching training dataset
            flow_data = {
                # Flow identifiers
                'id': flow.id,
                'expiration_id': flow.expiration_id,
                
                # Source information
                'src_ip': flow.src_ip,
                'src_mac': flow.src_mac,
                'src_oui': flow.src_oui,
                'src_port': flow.src_port,
                
                # Destination information
                'dst_ip': flow.dst_ip,
                'dst_mac': flow.dst_mac,
                'dst_oui': flow.dst_oui,
                'dst_port': flow.dst_port,
                
                # Protocol information
                'protocol': flow.protocol,
                'ip_version': flow.ip_version,
                'vlan_id': flow.vlan_id,
                'tunnel_id': flow.tunnel_id,
                
                # Bidirectional temporal features
                'bidirectional_first_seen_ms': flow.bidirectional_first_seen_ms,
                'bidirectional_last_seen_ms': flow.bidirectional_last_seen_ms,
                'bidirectional_duration_ms': flow.bidirectional_duration_ms,
                'bidirectional_packets': flow.bidirectional_packets,
                'bidirectional_bytes': flow.bidirectional_bytes,
                
                # Source to destination temporal features
                'src2dst_first_seen_ms': flow.src2dst_first_seen_ms,
                'src2dst_last_seen_ms': flow.src2dst_last_seen_ms,
                'src2dst_duration_ms': flow.src2dst_duration_ms,
                'src2dst_packets': flow.src2dst_packets,
                'src2dst_bytes': flow.src2dst_bytes,
                
                # Destination to source temporal features
                'dst2src_first_seen_ms': flow.dst2src_first_seen_ms,
                'dst2src_last_seen_ms': flow.dst2src_last_seen_ms,
                'dst2src_duration_ms': flow.dst2src_duration_ms,
                'dst2src_packets': flow.dst2src_packets,
                'dst2src_bytes': flow.dst2src_bytes,
                
                # Bidirectional packet size statistics
                'bidirectional_min_ps': flow.bidirectional_min_ps,
                'bidirectional_mean_ps': flow.bidirectional_mean_ps,
                'bidirectional_stddev_ps': flow.bidirectional_stddev_ps,
                'bidirectional_max_ps': flow.bidirectional_max_ps,
                
                # Source to destination packet size statistics
                'src2dst_min_ps': flow.src2dst_min_ps,
                'src2dst_mean_ps': flow.src2dst_mean_ps,
                'src2dst_stddev_ps': flow.src2dst_stddev_ps,
                'src2dst_max_ps': flow.src2dst_max_ps,
                
                # Destination to source packet size statistics
                'dst2src_min_ps': flow.dst2src_min_ps,
                'dst2src_mean_ps': flow.dst2src_mean_ps,
                'dst2src_stddev_ps': flow.dst2src_stddev_ps,
                'dst2src_max_ps': flow.dst2src_max_ps,
                
                # Bidirectional inter-arrival time statistics
                'bidirectional_min_piat_ms': flow.bidirectional_min_piat_ms,
                'bidirectional_mean_piat_ms': flow.bidirectional_mean_piat_ms,
                'bidirectional_stddev_piat_ms': flow.bidirectional_stddev_piat_ms,
                'bidirectional_max_piat_ms': flow.bidirectional_max_piat_ms,
                
                # Source to destination inter-arrival time statistics
                'src2dst_min_piat_ms': flow.src2dst_min_piat_ms,
                'src2dst_mean_piat_ms': flow.src2dst_mean_piat_ms,
                'src2dst_stddev_piat_ms': flow.src2dst_stddev_piat_ms,
                'src2dst_max_piat_ms': flow.src2dst_max_piat_ms,
                
                # Destination to source inter-arrival time statistics
                'dst2src_min_piat_ms': flow.dst2src_min_piat_ms,
                'dst2src_mean_piat_ms': flow.dst2src_mean_piat_ms,
                'dst2src_stddev_piat_ms': flow.dst2src_stddev_piat_ms,
                'dst2src_max_piat_ms': flow.dst2src_max_piat_ms,
                
                # Bidirectional TCP flags
                'bidirectional_syn_packets': flow.bidirectional_syn_packets,
                'bidirectional_cwr_packets': flow.bidirectional_cwr_packets,
                'bidirectional_ece_packets': flow.bidirectional_ece_packets,
                'bidirectional_urg_packets': flow.bidirectional_urg_packets,
                'bidirectional_ack_packets': flow.bidirectional_ack_packets,
                'bidirectional_psh_packets': flow.bidirectional_psh_packets,
                'bidirectional_rst_packets': flow.bidirectional_rst_packets,
                'bidirectional_fin_packets': flow.bidirectional_fin_packets,
                
                # Source to destination TCP flags
                'src2dst_syn_packets': flow.src2dst_syn_packets,
                'src2dst_cwr_packets': flow.src2dst_cwr_packets,
                'src2dst_ece_packets': flow.src2dst_ece_packets,
                'src2dst_urg_packets': flow.src2dst_urg_packets,
                'src2dst_ack_packets': flow.src2dst_ack_packets,
                'src2dst_psh_packets': flow.src2dst_psh_packets,
                'src2dst_rst_packets': flow.src2dst_rst_packets,
                'src2dst_fin_packets': flow.src2dst_fin_packets,
                
                # Destination to source TCP flags
                'dst2src_syn_packets': flow.dst2src_syn_packets,
                'dst2src_cwr_packets': flow.dst2src_cwr_packets,
                'dst2src_ece_packets': flow.dst2src_ece_packets,
                'dst2src_urg_packets': flow.dst2src_urg_packets,
                'dst2src_ack_packets': flow.dst2src_ack_packets,
                'dst2src_psh_packets': flow.dst2src_psh_packets,
                'dst2src_rst_packets': flow.dst2src_rst_packets,
                'dst2src_fin_packets': flow.dst2src_fin_packets,
                
                # Application layer information
                'application_name': flow.application_name,
                'application_category_name': flow.application_category_name,
                'application_is_guessed': flow.application_is_guessed,
                'application_confidence': flow.application_confidence,
                'requested_server_name': flow.requested_server_name,
                'client_fingerprint': flow.client_fingerprint,
                'server_fingerprint': flow.server_fingerprint,
                'user_agent': flow.user_agent,
                'content_type': flow.content_type,
            }
            
            # Add label if provided
            if label:
                flow_data['Label'] = label
            else:
                flow_data['Label'] = 'unknown'
            
            flows.append(flow_data)
            flow_count += 1
            
            # Stop if max_flows reached
            if max_flows and flow_count >= max_flows:
                logger.info(f"Reached maximum flow limit: {max_flows}")
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(flows)
        
        logger.info(f"✓ Extracted {len(df)} flows")
        logger.info(f"  Columns: {len(df.columns)}")
        
        # Save to CSV
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        
        logger.info(f"✓ Saved to: {output_file}")
        
        # Display summary statistics
        print_header("FLOW STATISTICS", width=60, char='-')
        print(f"Total flows: {len(df):,}")
        print(f"Total packets: {df['bidirectional_packets'].sum():,}")
        print(f"Total bytes: {df['bidirectional_bytes'].sum():,}")
        print(f"\nProtocol distribution:")
        print(df['protocol'].value_counts().head(10))
        
        if label:
            print(f"\nLabel: {label}")
            print(f"  Flows: {len(df[df['Label'] == label]):,}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing PCAP: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Convert PCAP to CSV for ARP Spoofing Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert PCAP to CSV (unlabeled)
  python scripts/pcap_to_csv.py capture.pcap output.csv
  
  # Convert with normal traffic label
  python scripts/pcap_to_csv.py normal_traffic.pcap normal.csv --label normal
  
  # Convert with attack label
  python scripts/pcap_to_csv.py attack.pcap attack.csv --label arp_spoofing
  
  # Process only first 1000 flows
  python scripts/pcap_to_csv.py large.pcap output.csv --max-flows 1000
  
  # Enable debug logging
  python scripts/pcap_to_csv.py input.pcap output.csv --verbose

Note: Requires nfstream library
  Install with: pip install nfstream
        """
    )
    
    parser.add_argument('pcap_file', type=str, help='Input PCAP file')
    parser.add_argument('output_file', type=str, help='Output CSV file')
    parser.add_argument(
        '--label', 
        type=str, 
        default=None,
        choices=['normal', 'arp_spoofing', 'Normal', 'MITM-ArpSpoofing', 'Attacker', 'unknown'],
        help='Label for the flows (default: unknown)'
    )
    parser.add_argument(
        '--max-flows',
        type=int,
        default=None,
        help='Maximum number of flows to process (default: all)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        level='DEBUG' if args.verbose else 'INFO'
    )
    
    print_header("PCAP TO CSV CONVERTER", width=60)
    
    # Check if nfstream is available
    if not NFSTREAM_AVAILABLE:
        print("\n❌ Error: nfstream library is not installed")
        print("\nInstall it with:")
        print("  pip install nfstream")
        print("\nOr add to requirements.txt:")
        print("  echo 'nfstream' >> requirements.txt")
        print("  pip install -r requirements.txt")
        return 1
    
    # Convert PCAP to CSV
    success = convert_pcap_to_csv(
        pcap_file=args.pcap_file,
        output_file=args.output_file,
        label=args.label,
        max_flows=args.max_flows
    )
    
    if success:
        print_header("CONVERSION COMPLETE", width=60)
        print(f"\n✓ CSV file ready: {args.output_file}")
        print("\nNext steps:")
        print("  1. Review the CSV file to verify features")
        print("  2. Use for real-time detection or batch analysis")
        print("  3. Upload to Flask web interface (/analyze)")
        return 0
    else:
        print_header("CONVERSION FAILED", width=60)
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nConversion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)
