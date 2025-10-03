# P9ML Performance Benchmarks

This directory contains comprehensive benchmarking tools to evaluate the performance of traditional neural networks versus P9ML Membrane Computing enhanced networks.

## Files Overview

- **`P9MLPerformanceBenchmark.lua`**: Complete Lua/Torch benchmark suite
- **`p9ml_performance_benchmark.py`**: Python simulation equivalent for non-Torch environments  
- **`IndexLinear.lua`**: Existing IndexLinear benchmark (original nn package)

## Quick Start

### Lua/Torch Environment (Recommended)

Run complete benchmark suite:
```bash
th run_p9ml_benchmark.lua
```

Run quick test mode:
```bash
th run_p9ml_benchmark.lua --quick
```

Integrate with test suite:
```lua
th test.lua P9MLPerformanceBenchmark
```

### Python Environment (Simulation)

For environments without Torch:
```bash
python test/benchmarks/p9ml_performance_benchmark.py
```

## Benchmark Metrics

### Memory Efficiency
- Forward pass memory allocation
- Backward pass memory allocation  
- Total memory overhead comparison

### Training Performance
- Total training time across epochs
- Average time per batch
- Loss convergence comparison

### Inference Speed
- Forward pass execution time
- Throughput (samples/second)
- Statistical analysis (min/max/mean)

## Network Configurations

| Size | Architecture | Description |
|------|-------------|-------------|
| Small | 10→32→16→2 | Lightweight networks |
| Medium | 100→128→64→32→10 | Standard classification |
| Large | 784→256→128→64→10 | Complex image tasks |

## P9ML Enhancements

Each enhanced network includes:
- Membrane-wrapped layers
- Evolution rules (gradient, quantization, cognitive)
- Namespace coordination
- Cognitive kernel with lexeme management
- Meta-learning capabilities

## Output

Results are exported to CSV format for analysis:
- Detailed performance metrics
- Statistical comparisons
- Recommendations for optimization

See `../docs/performance_baseline.md` for comprehensive documentation.

## Integration

The benchmark system integrates with:
- nn.test framework
- P9ML test suite
- Continuous integration pipelines
- Performance regression tracking

## Troubleshooting

Common issues:
- Ensure all P9ML components are loaded
- Check available memory for large configurations
- Use `--quick` mode for rapid testing
- Monitor system load during benchmarks

For detailed troubleshooting, see the full documentation.