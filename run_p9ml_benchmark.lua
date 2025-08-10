#!/usr/bin/env lua
-- P9ML Performance Benchmark Runner
-- Executes the complete performance benchmark suite comparing traditional vs P9ML networks

require('torch')
require('nn')

-- Load the benchmark suite
local P9MLPerformanceBenchmark = require('test.benchmarks.P9MLPerformanceBenchmark')

print("="*80)
print("P9ML Performance Benchmark Suite Runner")
print("="*80)
print("")

-- System information
print("System Information:")
print(string.format("  Torch Version: %s", torch._VERSION or "unknown"))
print(string.format("  Platform: %s", jit and jit.os or "unknown"))
print(string.format("  Architecture: %s", jit and jit.arch or "unknown"))

-- Check for GPU availability  
local gpuAvailable = pcall(require, 'cunn')
if gpuAvailable then
    local cudnn = pcall(require, 'cudnn')
    print(string.format("  GPU: Available %s", cudnn and "(with CuDNN)" or "(without CuDNN)"))
else
    print("  GPU: Not available (CPU only)")
end

-- Memory information
print(string.format("  Available Memory: %.1f MB", collectgarbage("count")))
print("")

-- Check if P9ML components are available
local function checkP9MLComponents()
    local components = {
        'P9MLMembrane',
        'P9MLNamespace', 
        'P9MLCognitiveKernel',
        'P9MLEvolution'
    }
    
    print("P9ML Component Check:")
    local all_available = true
    
    for _, component in ipairs(components) do
        local available, _ = pcall(function() return nn[component] end)
        print(string.format("  %s: %s", component, available and "✓ Available" or "✗ Missing"))
        if not available then
            all_available = false
        end
    end
    
    return all_available
end

local p9ml_available = checkP9MLComponents()
print("")

if not p9ml_available then
    print("ERROR: P9ML components are not fully available.")
    print("Please ensure all P9ML modules are properly loaded.")
    print("Run with: th -lnn -e \"require('P9MLTest').checkComponents()\"")
    os.exit(1)
end

-- Benchmark configuration override for quick testing
local quick_test = false
if arg and arg[1] == "--quick" then
    quick_test = true
    print("Quick test mode enabled - reduced test scope")
    print("")
end

-- Run the benchmark suite
print("Starting Performance Benchmark Suite...")
print("")

local start_time = os.clock()

-- Run benchmarks
local results
if quick_test then
    -- Quick test with limited configurations
    local quick_config = {
        architectures = {
            small = {
                layers = {10, 32, 2},
                name = "Quick Test Network (10->32->2)"
            }
        },
        batch_sizes = {16},
        num_epochs = 3,
        num_warmup_runs = 1,
        num_benchmark_runs = 3
    }
    
    -- Override default configuration
    P9MLPerformanceBenchmark.BenchmarkConfig = quick_config
    results = P9MLPerformanceBenchmark.runFullBenchmarkSuite()
else
    -- Full benchmark suite
    results = P9MLPerformanceBenchmark.runFullBenchmarkSuite()
end

local end_time = os.clock()
local total_duration = end_time - start_time

print("")
print("="*80)
print("BENCHMARK SUITE COMPLETED")
print("="*80)
print(string.format("Total execution time: %.2f seconds", total_duration))
print(string.format("Configurations tested: %d", #results))
print("")

-- Export results
local timestamp = os.date("%Y%m%d_%H%M%S")
local csv_filename = string.format("p9ml_benchmark_%s.csv", timestamp)

P9MLPerformanceBenchmark.exportResultsToCSV(results, csv_filename)
print(string.format("Detailed results exported to: %s", csv_filename))

-- Generate summary statistics
local function calculateSummaryStats(results)
    local memory_improvements = {}
    local training_ratios = {}
    local inference_ratios = {}
    
    for _, result in ipairs(results) do
        local mem_improve = (result.traditional.memory.total - result.p9ml.memory.total) / result.traditional.memory.total
        table.insert(memory_improvements, mem_improve)
        
        local train_ratio = result.traditional.training.total_training_time / result.p9ml.training.total_training_time  
        table.insert(training_ratios, train_ratio)
        
        local inf_ratio = result.traditional.inference.avg_inference_time / result.p9ml.inference.avg_inference_time
        table.insert(inference_ratios, inf_ratio)
    end
    
    local function calculateMean(values)
        local sum = 0
        for _, v in ipairs(values) do
            sum = sum + v
        end
        return sum / #values
    end
    
    return {
        avg_memory_improvement = calculateMean(memory_improvements),
        avg_training_speedup = calculateMean(training_ratios),
        avg_inference_speedup = calculateMean(inference_ratios)
    }
end

local summary = calculateSummaryStats(results)

print("")
print("SUMMARY STATISTICS:")
print(string.format("  Average memory improvement: %.1f%% %s", 
                   math.abs(summary.avg_memory_improvement) * 100,
                   summary.avg_memory_improvement > 0 and "reduction" or "increase"))
print(string.format("  Average training speedup: %.2fx", summary.avg_training_speedup))
print(string.format("  Average inference speedup: %.2fx", summary.avg_inference_speedup))

-- Performance recommendations
print("")
print("PERFORMANCE RECOMMENDATIONS:")

if summary.avg_memory_improvement > 0.1 then
    print("  ✓ P9ML shows significant memory efficiency gains")
elseif summary.avg_memory_improvement < -0.1 then
    print("  ⚠ P9ML has higher memory overhead - consider optimization")
else
    print("  ~ P9ML memory usage is comparable to traditional networks")
end

if summary.avg_training_speedup > 1.1 then
    print("  ✓ P9ML training is faster than traditional networks")
elseif summary.avg_training_speedup < 0.9 then
    print("  ⚠ P9ML training is slower - evolution rules may need tuning")
else
    print("  ~ P9ML training speed is comparable to traditional networks")
end

if summary.avg_inference_speedup > 1.1 then
    print("  ✓ P9ML inference is faster than traditional networks")
elseif summary.avg_inference_speedup < 0.9 then
    print("  ⚠ P9ML inference is slower - consider quantization optimization")
else
    print("  ~ P9ML inference speed is comparable to traditional networks")
end

print("")
print("For detailed analysis, examine the exported CSV file:")
print(string.format("  %s", csv_filename))
print("")
print("To run a quick test: th run_p9ml_benchmark.lua --quick")
print("="*80)