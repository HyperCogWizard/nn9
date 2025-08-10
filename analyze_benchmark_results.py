#!/usr/bin/env python3
"""
P9ML Benchmark Results Analyzer
Analyzes the CSV results and provides insights into performance characteristics.
"""

import csv
import sys
from pathlib import Path

def analyze_benchmark_results(csv_file="p9ml_benchmark_results.csv"):
    """Analyze benchmark results from CSV file"""
    
    if not Path(csv_file).exists():
        print(f"Error: {csv_file} not found. Run benchmarks first.")
        return
    
    results = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'Architecture': row['Architecture'],
                'BatchSize': int(row['BatchSize']),
                'NetworkType': row['NetworkType'],
                'MemoryTotal': float(row['MemoryTotal']),
                'TrainingTime': float(row['TrainingTime']),
                'Throughput': float(row['Throughput']),
                'FinalLoss': float(row['FinalLoss'])
            })
    
    print("P9ML Benchmark Results Analysis")
    print("=" * 50)
    
    # Group results by architecture and batch size
    configs = {}
    for result in results:
        key = (result['Architecture'], result['BatchSize'])
        if key not in configs:
            configs[key] = {}
        configs[key][result['NetworkType']] = result
    
    print("\nPerformance Comparison Summary")
    print("-" * 50)
    print(f"{'Architecture':<25} {'Batch':<5} {'Mem Overhead':<12} {'Train Speed':<12} {'Inference':<12}")
    print(f"{'':25} {'Size':<5} {'(P9ML/Trad)':<12} {'(Trad/P9ML)':<12} {'(P9ML/Trad)':<12}")
    print("-" * 70)
    
    memory_overheads = []
    training_ratios = []
    inference_ratios = []
    
    for (arch, batch), data in sorted(configs.items()):
        if 'Traditional' in data and 'P9ML' in data:
            trad = data['Traditional']
            p9ml = data['P9ML']
            
            # Calculate metrics
            memory_overhead = p9ml['MemoryTotal'] / trad['MemoryTotal']
            training_ratio = trad['TrainingTime'] / p9ml['TrainingTime'] 
            inference_ratio = p9ml['Throughput'] / trad['Throughput']
            
            # Store for overall analysis
            memory_overheads.append(memory_overhead)
            training_ratios.append(training_ratio)
            inference_ratios.append(inference_ratio)
            
            # Display
            arch_short = arch.split('(')[0].strip()
            print(f"{arch_short:<25} {batch:<5} {memory_overhead:<12.2f} {training_ratio:<12.2f} {inference_ratio:<12.2f}")
    
    print("\nOverall Statistics")
    print("-" * 50)
    
    avg_memory_overhead = sum(memory_overheads) / len(memory_overheads)
    avg_training_ratio = sum(training_ratios) / len(training_ratios) 
    avg_inference_ratio = sum(inference_ratios) / len(inference_ratios)
    
    print(f"Average Memory Overhead: {avg_memory_overhead:.2f}x ({(avg_memory_overhead-1)*100:.1f}% increase)")
    print(f"Average Training Speed: {avg_training_ratio:.2f}x ({'P9ML faster' if avg_training_ratio > 1 else 'Traditional faster'})")
    print(f"Average Inference Speed: {avg_inference_ratio:.2f}x ({'P9ML faster' if avg_inference_ratio > 1 else 'Traditional faster'})")
    
    print("\nDetailed Analysis by Architecture")
    print("-" * 50)
    
    architectures = set(result['Architecture'] for result in results)
    
    for arch in sorted(architectures):
        print(f"\n{arch}:")
        
        arch_results = [r for r in results if r['Architecture'] == arch]
        trad_results = [r for r in arch_results if r['NetworkType'] == 'Traditional']
        p9ml_results = [r for r in arch_results if r['NetworkType'] == 'P9ML']
        
        if trad_results and p9ml_results:
            # Memory analysis
            trad_mem_range = (min(r['MemoryTotal'] for r in trad_results), 
                             max(r['MemoryTotal'] for r in trad_results))
            p9ml_mem_range = (min(r['MemoryTotal'] for r in p9ml_results),
                             max(r['MemoryTotal'] for r in p9ml_results))
            
            print(f"  Memory Usage:")
            print(f"    Traditional: {trad_mem_range[0]:.2f} - {trad_mem_range[1]:.2f} MB")
            print(f"    P9ML:        {p9ml_mem_range[0]:.2f} - {p9ml_mem_range[1]:.2f} MB")
            
            # Training analysis
            trad_train = trad_results[0]['TrainingTime']  # Same across batch sizes
            p9ml_train = p9ml_results[0]['TrainingTime']
            print(f"  Training Time: Traditional {trad_train:.3f}s vs P9ML {p9ml_train:.3f}s")
            
            # Inference analysis
            trad_throughput = [r['Throughput'] for r in trad_results]
            p9ml_throughput = [r['Throughput'] for r in p9ml_results]
            
            print(f"  Throughput Range:")
            print(f"    Traditional: {min(trad_throughput):,.0f} - {max(trad_throughput):,.0f} samples/sec")
            print(f"    P9ML:        {min(p9ml_throughput):,.0f} - {max(p9ml_throughput):,.0f} samples/sec")
            
            # Loss comparison
            trad_loss = [r['FinalLoss'] for r in trad_results]
            p9ml_loss = [r['FinalLoss'] for r in p9ml_results]
            avg_trad_loss = sum(trad_loss) / len(trad_loss)
            avg_p9ml_loss = sum(p9ml_loss) / len(p9ml_loss)
            
            print(f"  Average Final Loss: Traditional {avg_trad_loss:.6f} vs P9ML {avg_p9ml_loss:.6f}")
    
    print("\nKey Insights")
    print("-" * 50)
    
    if avg_memory_overhead > 1.3:
        print("• P9ML has significant memory overhead - consider optimization for memory-constrained environments")
    elif avg_memory_overhead > 1.1:
        print("• P9ML has moderate memory overhead - acceptable for most applications")
    else:
        print("• P9ML memory usage is comparable to traditional networks")
    
    if avg_training_ratio > 1.1:
        print("• P9ML trains faster than traditional networks - evolution rules are effective")
    elif avg_training_ratio < 0.9:
        print("• P9ML trains slower than traditional networks - overhead from cognitive processing")
    else:
        print("• P9ML training speed is comparable to traditional networks")
    
    if avg_inference_ratio > 1.1:
        print("• P9ML inference is faster than traditional networks - quantization benefits")
    elif avg_inference_ratio < 0.9:
        print("• P9ML inference is slower than traditional networks - membrane overhead")  
    else:
        print("• P9ML inference speed is comparable to traditional networks")
    
    print("\nRecommendations")
    print("-" * 50)
    
    if avg_memory_overhead < 1.5 and avg_training_ratio > 0.7 and avg_inference_ratio > 0.9:
        print("• P9ML shows good overall performance - suitable for production use")
        print("• Consider P9ML for applications requiring adaptive/cognitive capabilities")
    else:
        print("• P9ML has performance trade-offs - evaluate based on specific requirements")
        print("• Traditional networks may be better for simple, performance-critical tasks")
    
    if avg_training_ratio < 0.8:
        print("• Training overhead is significant - consider meta-learning benefits for long training")
    
    if avg_inference_ratio > 0.95:
        print("• Inference performance is well-maintained - suitable for real-time applications")

def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "p9ml_benchmark_results.csv"
    analyze_benchmark_results(csv_file)

if __name__ == "__main__":
    main()