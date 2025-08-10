-- P9ML Performance Benchmark Integration with Test Suite
-- Adds performance benchmarking to the nn test framework

local P9MLPerformanceBenchmarkTest = {}

-- Integration with nn.test framework
function P9MLPerformanceBenchmarkTest.P9MLPerformanceBenchmark()
    local mytester = torch.Tester()
    
    -- Load the benchmark module
    local success, P9MLPerformanceBenchmark = pcall(require, 'test.benchmarks.P9MLPerformanceBenchmark')
    mytester:assert(success, "P9ML Performance Benchmark module should load successfully")
    
    if not success then
        return
    end
    
    -- Test benchmark configuration
    mytester:assert(P9MLPerformanceBenchmark.BenchmarkConfig, "Benchmark config should be available")
    mytester:assert(P9MLPerformanceBenchmark.BenchmarkConfig.architectures, "Architectures should be defined")
    mytester:assert(P9MLPerformanceBenchmark.BenchmarkConfig.batch_sizes, "Batch sizes should be defined")
    
    -- Test network creation functions
    print("Testing traditional network creation...")
    local layers = {10, 32, 16, 2}
    local traditional_net = P9MLPerformanceBenchmark.createTraditionalNetwork(layers)
    mytester:assert(traditional_net, "Traditional network should be created")
    mytester:assert(torch.type(traditional_net) == 'nn.Sequential', "Traditional network should be Sequential")
    
    print("Testing P9ML network creation...")
    local p9ml_net, namespace, kernel, membranes = P9MLPerformanceBenchmark.createP9MLNetwork(layers)
    mytester:assert(p9ml_net, "P9ML network should be created")
    mytester:assert(torch.type(p9ml_net) == 'nn.Sequential', "P9ML network should be Sequential")
    mytester:assert(namespace, "Namespace should be created")
    mytester:assert(kernel, "Cognitive kernel should be created")
    mytester:assert(membranes, "Membranes should be created")
    
    -- Test memory measurement
    print("Testing memory measurement...")
    local test_input = torch.randn(4, layers[1])
    local memory_stats = P9MLPerformanceBenchmark.measureMemoryUsage(traditional_net, test_input)
    mytester:assert(memory_stats, "Memory stats should be returned")
    mytester:assert(memory_stats.baseline, "Baseline memory should be measured")
    mytester:assert(memory_stats.forward, "Forward memory should be measured")
    mytester:assert(memory_stats.backward, "Backward memory should be measured")
    mytester:assert(memory_stats.total, "Total memory should be calculated")
    
    -- Test quick benchmark run
    print("Running quick performance benchmark...")
    local quick_config = {
        architectures = {
            tiny = {
                layers = {5, 8, 3},
                name = "Tiny Test Network (5->8->3)"
            }
        },
        batch_sizes = {4},
        num_epochs = 2,
        num_warmup_runs = 1,
        num_benchmark_runs = 2
    }
    
    -- Override config temporarily
    local original_config = P9MLPerformanceBenchmark.BenchmarkConfig
    P9MLPerformanceBenchmark.BenchmarkConfig = quick_config
    
    local result = P9MLPerformanceBenchmark.benchmarkNetwork('tiny', quick_config.architectures.tiny, 4)
    mytester:assert(result, "Benchmark result should be returned")
    mytester:assert(result.architecture == 'tiny', "Architecture should match")
    mytester:assert(result.batch_size == 4, "Batch size should match")
    mytester:assert(result.traditional, "Traditional results should be present")
    mytester:assert(result.p9ml, "P9ML results should be present")
    mytester:assert(result.comparison, "Comparison metrics should be present")
    
    -- Restore original config
    P9MLPerformanceBenchmark.BenchmarkConfig = original_config
    
    -- Test report generation
    print("Testing report generation...")
    local success_report = pcall(P9MLPerformanceBenchmark.generatePerformanceReport, {result})
    mytester:assert(success_report, "Performance report should generate successfully")
    
    -- Test CSV export
    print("Testing CSV export...")
    local test_filename = "/tmp/test_benchmark_results.csv"
    local success_csv = pcall(P9MLPerformanceBenchmark.exportResultsToCSV, {result}, test_filename)
    mytester:assert(success_csv, "CSV export should succeed")
    
    -- Verify CSV file was created
    local file = io.open(test_filename, "r")
    if file then
        local content = file:read("*all")
        file:close()
        mytester:assert(content and #content > 0, "CSV file should have content")
        os.remove(test_filename)  -- Clean up
    end
    
    print("✓ P9ML Performance Benchmark tests completed successfully")
end

-- Standalone test runner for performance benchmarks
function P9MLPerformanceBenchmarkTest.runQuickBenchmark()
    print("Running quick P9ML performance benchmark...")
    
    local P9MLPerformanceBenchmark = require('test.benchmarks.P9MLPerformanceBenchmark')
    
    -- Configure for quick test
    local quick_config = {
        architectures = {
            small = {
                layers = {10, 16, 8, 2},
                name = "Quick Test Network (10->16->8->2)"
            }
        },
        batch_sizes = {8, 32},
        num_epochs = 3,
        num_warmup_runs = 1,
        num_benchmark_runs = 3
    }
    
    -- Override configuration
    P9MLPerformanceBenchmark.BenchmarkConfig = quick_config
    
    -- Run benchmark
    local results = P9MLPerformanceBenchmark.runFullBenchmarkSuite()
    
    print(string.format("Quick benchmark completed: %d configurations tested", #results))
    return results
end

return P9MLPerformanceBenchmarkTest