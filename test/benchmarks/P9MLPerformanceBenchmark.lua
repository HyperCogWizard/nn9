-- P9ML Performance Benchmark Suite
-- Compares traditional neural networks with P9ML enhanced networks
-- Measures memory usage, training time, inference speed, and scalability

require('torch')
require('nn')
require('sys')

local P9MLPerformanceBenchmark = {}

-- Benchmark configuration
local BenchmarkConfig = {
    -- Network architectures to test
    architectures = {
        small = {
            layers = {10, 32, 16, 2},
            name = "Small Network (10->32->16->2)"
        },
        medium = {
            layers = {100, 128, 64, 32, 10},
            name = "Medium Network (100->128->64->32->10)"
        },
        large = {
            layers = {784, 256, 128, 64, 10},
            name = "Large Network (784->256->128->64->10)"
        }
    },
    
    -- Benchmark parameters
    batch_sizes = {1, 16, 64, 256},
    num_epochs = 10,
    num_warmup_runs = 3,
    num_benchmark_runs = 10,
    
    -- Memory measurement intervals
    memory_sample_interval = 100,
    
    -- Results storage
    results = {}
}

-- Traditional neural network factory
function P9MLPerformanceBenchmark.createTraditionalNetwork(layers)
    local net = nn.Sequential()
    
    for i = 1, #layers - 1 do
        net:add(nn.Linear(layers[i], layers[i + 1]))
        if i < #layers - 1 then
            net:add(nn.ReLU())
        end
    end
    
    net:add(nn.LogSoftMax())
    return net
end

-- P9ML enhanced network factory
function P9MLPerformanceBenchmark.createP9MLNetwork(layers)
    local net = nn.Sequential()
    local membranes = {}
    
    for i = 1, #layers - 1 do
        local linear = nn.Linear(layers[i], layers[i + 1])
        local membrane_id = string.format("layer_%d", i)
        local membrane = nn.P9MLMembrane(linear, membrane_id)
        
        -- Add evolution rules based on layer position
        if i == 1 then
            -- Input layer: gradient evolution
            membrane:addEvolutionRule(nn.P9MLEvolutionFactory.createGradientEvolution(0.01, 0.9))
        elseif i == #layers - 1 then
            -- Output layer: cognitive adaptation
            membrane:addEvolutionRule(nn.P9MLEvolutionFactory.createCognitiveAdaptation(0.01, 0.9))
        else
            -- Hidden layers: adaptive quantization
            membrane:addEvolutionRule(nn.P9MLEvolutionFactory.createAdaptiveQuantization(8, 0.1))
            membrane:enableQuantization(8, 0.1)
        end
        
        net:add(membrane)
        membranes[i] = membrane
        
        if i < #layers - 1 then
            net:add(nn.ReLU())
        end
    end
    
    net:add(nn.LogSoftMax())
    
    -- Create namespace for coordination
    local namespace = nn.P9MLNamespace("benchmark_network")
    for i, membrane in ipairs(membranes) do
        namespace:registerMembrane(membrane, string.format("layer_%d", i))
    end
    
    -- Create cognitive kernel
    local kernel = nn.P9MLCognitiveKernel()
    for i = 1, #layers - 1 do
        kernel:addLexeme({layers[i], layers[i + 1]}, string.format("transformation_%d", i))
    end
    
    return net, namespace, kernel, membranes
end

-- Memory usage measurement
function P9MLPerformanceBenchmark.measureMemoryUsage(network, input_batch)
    -- Force garbage collection before measurement
    collectgarbage("collect")
    
    local memory_before = collectgarbage("count")
    
    -- Forward pass
    local output = network:forward(input_batch)
    
    local memory_after_forward = collectgarbage("count")
    
    -- Backward pass
    local grad_output = torch.randn(output:size())
    local grad_input = network:backward(input_batch, grad_output)
    
    local memory_after_backward = collectgarbage("count")
    
    return {
        baseline = memory_before,
        forward = memory_after_forward - memory_before,
        backward = memory_after_backward - memory_after_forward,
        total = memory_after_backward - memory_before
    }
end

-- Training time benchmark
function P9MLPerformanceBenchmark.benchmarkTrainingTime(network, architecture, batch_size, namespace, kernel)
    local input_size = architecture.layers[1]
    local output_size = architecture.layers[#architecture.layers]
    
    -- Generate synthetic training data
    local train_data = {}
    local train_targets = {}
    local num_batches = 50
    
    for i = 1, num_batches do
        train_data[i] = torch.randn(batch_size, input_size)
        train_targets[i] = torch.randperm(output_size):narrow(1, 1, batch_size):long()
    end
    
    local criterion = nn.ClassNLLCriterion()
    local optimizer = {
        learningRate = 0.01,
        momentum = 0.9
    }
    
    -- Warmup runs
    for i = 1, BenchmarkConfig.num_warmup_runs do
        local output = network:forward(train_data[1])
        local loss = criterion:forward(output, train_targets[1])
        local grad_output = criterion:backward(output, train_targets[1])
        network:backward(train_data[1], grad_output)
    end
    
    -- Benchmark training
    local training_times = {}
    local epoch_losses = {}
    
    for epoch = 1, BenchmarkConfig.num_epochs do
        local epoch_start = sys.clock()
        local epoch_loss = 0
        
        for batch = 1, num_batches do
            local batch_start = sys.clock()
            
            -- Forward pass
            local output = network:forward(train_data[batch])
            local loss = criterion:forward(output, train_targets[batch])
            epoch_loss = epoch_loss + loss
            
            -- Backward pass
            local grad_output = criterion:backward(output, train_targets[batch])
            network:backward(train_data[batch], grad_output)
            
            -- Apply P9ML specific operations if available
            if namespace then
                if batch % 5 == 0 then
                    namespace:applyMetaLearning()
                end
            end
            
            if kernel and batch % 10 == 0 then
                kernel:generateGestaltField()
            end
            
            local batch_time = sys.clock() - batch_start
            table.insert(training_times, batch_time)
        end
        
        local epoch_time = sys.clock() - epoch_start
        epoch_losses[epoch] = epoch_loss / num_batches
        
        print(string.format("Epoch %d: Loss=%.6f, Time=%.3fs", epoch, epoch_losses[epoch], epoch_time))
    end
    
    -- Calculate statistics
    local total_time = 0
    for _, time in ipairs(training_times) do
        total_time = total_time + time
    end
    
    local avg_batch_time = total_time / #training_times
    local final_loss = epoch_losses[#epoch_losses]
    
    return {
        total_training_time = total_time,
        avg_batch_time = avg_batch_time,
        final_loss = final_loss,
        epoch_losses = epoch_losses,
        batch_times = training_times
    }
end

-- Inference speed benchmark
function P9MLPerformanceBenchmark.benchmarkInferenceSpeed(network, architecture, batch_size)
    local input_size = architecture.layers[1]
    
    -- Generate test data
    local test_inputs = {}
    for i = 1, BenchmarkConfig.num_benchmark_runs do
        test_inputs[i] = torch.randn(batch_size, input_size)
    end
    
    -- Warmup runs
    for i = 1, BenchmarkConfig.num_warmup_runs do
        network:forward(test_inputs[1])
    end
    
    -- Benchmark inference
    local inference_times = {}
    
    for i = 1, BenchmarkConfig.num_benchmark_runs do
        local start_time = sys.clock()
        local output = network:forward(test_inputs[i])
        local end_time = sys.clock()
        
        table.insert(inference_times, end_time - start_time)
    end
    
    -- Calculate statistics
    local total_time = 0
    local min_time = inference_times[1]
    local max_time = inference_times[1]
    
    for _, time in ipairs(inference_times) do
        total_time = total_time + time
        min_time = math.min(min_time, time)
        max_time = math.max(max_time, time)
    end
    
    local avg_time = total_time / #inference_times
    local throughput = batch_size / avg_time  -- samples per second
    
    return {
        avg_inference_time = avg_time,
        min_inference_time = min_time,
        max_inference_time = max_time,
        throughput = throughput,
        inference_times = inference_times
    }
end

-- Complete network benchmark
function P9MLPerformanceBenchmark.benchmarkNetwork(arch_name, architecture, batch_size)
    print(string.format("\n=== Benchmarking %s with batch size %d ===", architecture.name, batch_size))
    
    -- Create networks
    print("Creating traditional network...")
    local traditional_net = P9MLPerformanceBenchmark.createTraditionalNetwork(architecture.layers)
    
    print("Creating P9ML enhanced network...")
    local p9ml_net, namespace, kernel, membranes = P9MLPerformanceBenchmark.createP9MLNetwork(architecture.layers)
    
    -- Generate test input for memory measurement
    local test_input = torch.randn(batch_size, architecture.layers[1])
    
    -- Memory usage benchmarks
    print("\nMeasuring memory usage...")
    local traditional_memory = P9MLPerformanceBenchmark.measureMemoryUsage(traditional_net, test_input)
    local p9ml_memory = P9MLPerformanceBenchmark.measureMemoryUsage(p9ml_net, test_input)
    
    -- Training time benchmarks
    print("\nBenchmarking training time...")
    print("Traditional network training:")
    local traditional_training = P9MLPerformanceBenchmark.benchmarkTrainingTime(
        traditional_net, architecture, batch_size, nil, nil)
    
    print("P9ML enhanced network training:")
    local p9ml_training = P9MLPerformanceBenchmark.benchmarkTrainingTime(
        p9ml_net, architecture, batch_size, namespace, kernel)
    
    -- Inference speed benchmarks
    print("\nBenchmarking inference speed...")
    print("Traditional network inference:")
    local traditional_inference = P9MLPerformanceBenchmark.benchmarkInferenceSpeed(
        traditional_net, architecture, batch_size)
    
    print("P9ML enhanced network inference:")
    local p9ml_inference = P9MLPerformanceBenchmark.benchmarkInferenceSpeed(
        p9ml_net, architecture, batch_size)
    
    -- Compile results
    local results = {
        architecture = arch_name,
        batch_size = batch_size,
        traditional = {
            memory = traditional_memory,
            training = traditional_training,
            inference = traditional_inference
        },
        p9ml = {
            memory = p9ml_memory,
            training = p9ml_training,
            inference = p9ml_inference
        },
        comparison = {
            memory_improvement = (traditional_memory.total - p9ml_memory.total) / traditional_memory.total,
            training_speed_ratio = traditional_training.total_training_time / p9ml_training.total_training_time,
            inference_speed_ratio = traditional_inference.avg_inference_time / p9ml_inference.avg_inference_time,
            accuracy_difference = traditional_training.final_loss - p9ml_training.final_loss
        }
    }
    
    return results
end

-- Generate performance report
function P9MLPerformanceBenchmark.generatePerformanceReport(results)
    print("\n" .. string.rep("=", 80))
    print("P9ML PERFORMANCE BENCHMARK REPORT")
    print(string.rep("=", 80))
    
    for _, result in ipairs(results) do
        print(string.format("\n%s (Batch Size: %d)", 
                           BenchmarkConfig.architectures[result.architecture].name, 
                           result.batch_size))
        print(string.rep("-", 60))
        
        -- Memory usage comparison
        print("Memory Usage:")
        print(string.format("  Traditional: %.2f KB total (%.2f KB forward, %.2f KB backward)", 
                           result.traditional.memory.total,
                           result.traditional.memory.forward, 
                           result.traditional.memory.backward))
        print(string.format("  P9ML:        %.2f KB total (%.2f KB forward, %.2f KB backward)", 
                           result.p9ml.memory.total,
                           result.p9ml.memory.forward, 
                           result.p9ml.memory.backward))
        print(string.format("  Improvement: %.1f%% %s", 
                           math.abs(result.comparison.memory_improvement) * 100,
                           result.comparison.memory_improvement > 0 and "reduction" or "increase"))
        
        -- Training time comparison
        print("\nTraining Performance:")
        print(string.format("  Traditional: %.3fs total, %.4fs/batch, final loss: %.6f", 
                           result.traditional.training.total_training_time,
                           result.traditional.training.avg_batch_time,
                           result.traditional.training.final_loss))
        print(string.format("  P9ML:        %.3fs total, %.4fs/batch, final loss: %.6f", 
                           result.p9ml.training.total_training_time,
                           result.p9ml.training.avg_batch_time,
                           result.p9ml.training.final_loss))
        print(string.format("  Speed ratio: %.2fx %s", 
                           result.comparison.training_speed_ratio,
                           result.comparison.training_speed_ratio > 1 and "(P9ML faster)" or "(Traditional faster)"))
        
        -- Inference speed comparison
        print("\nInference Performance:")
        print(string.format("  Traditional: %.4fs avg, %.1f samples/sec", 
                           result.traditional.inference.avg_inference_time,
                           result.traditional.inference.throughput))
        print(string.format("  P9ML:        %.4fs avg, %.1f samples/sec", 
                           result.p9ml.inference.avg_inference_time,
                           result.p9ml.inference.throughput))
        print(string.format("  Speed ratio: %.2fx %s", 
                           result.comparison.inference_speed_ratio,
                           result.comparison.inference_speed_ratio > 1 and "(P9ML faster)" or "(Traditional faster)"))
        
        -- Overall assessment
        print("\nOverall Assessment:")
        local memory_verdict = result.comparison.memory_improvement > 0.05 and "Better" or 
                              result.comparison.memory_improvement < -0.05 and "Worse" or "Similar"
        local training_verdict = result.comparison.training_speed_ratio > 1.05 and "Faster" or 
                                result.comparison.training_speed_ratio < 0.95 and "Slower" or "Similar"  
        local inference_verdict = result.comparison.inference_speed_ratio > 1.05 and "Faster" or 
                                 result.comparison.inference_speed_ratio < 0.95 and "Slower" or "Similar"
        
        print(string.format("  Memory efficiency: %s", memory_verdict))
        print(string.format("  Training speed: %s", training_verdict))
        print(string.format("  Inference speed: %s", inference_verdict))
    end
    
    print(string.rep("=", 80))
end

-- Main benchmark runner
function P9MLPerformanceBenchmark.runFullBenchmarkSuite()
    print("P9ML Performance Benchmark Suite")
    print("Comparing traditional neural networks with P9ML enhanced networks")
    print(string.format("Torch version: %s", torch._VERSION or "unknown"))
    print(string.format("Test configurations: %d architectures, %d batch sizes", 
                       #BenchmarkConfig.architectures, #BenchmarkConfig.batch_sizes))
    
    local all_results = {}
    
    for arch_name, architecture in pairs(BenchmarkConfig.architectures) do
        for _, batch_size in ipairs(BenchmarkConfig.batch_sizes) do
            local result = P9MLPerformanceBenchmark.benchmarkNetwork(arch_name, architecture, batch_size)
            table.insert(all_results, result)
            
            -- Save intermediate results
            BenchmarkConfig.results[string.format("%s_batch_%d", arch_name, batch_size)] = result
        end
    end
    
    -- Generate final report
    P9MLPerformanceBenchmark.generatePerformanceReport(all_results)
    
    return all_results
end

-- Export results to CSV for external analysis
function P9MLPerformanceBenchmark.exportResultsToCSV(results, filename)
    filename = filename or "p9ml_benchmark_results.csv"
    
    local file = io.open(filename, "w")
    if not file then
        error("Could not open file for writing: " .. filename)
    end
    
    -- CSV header
    file:write("Architecture,BatchSize,NetworkType,MemoryTotal,MemoryForward,MemoryBackward,")
    file:write("TrainingTime,AvgBatchTime,FinalLoss,InferenceTime,Throughput\n")
    
    for _, result in ipairs(results) do
        local arch_name = BenchmarkConfig.architectures[result.architecture].name
        
        -- Traditional network row
        file:write(string.format("%s,%d,Traditional,%.2f,%.2f,%.2f,%.3f,%.4f,%.6f,%.4f,%.1f\n",
                                arch_name, result.batch_size,
                                result.traditional.memory.total,
                                result.traditional.memory.forward,
                                result.traditional.memory.backward,
                                result.traditional.training.total_training_time,
                                result.traditional.training.avg_batch_time,
                                result.traditional.training.final_loss,
                                result.traditional.inference.avg_inference_time,
                                result.traditional.inference.throughput))
        
        -- P9ML network row
        file:write(string.format("%s,%d,P9ML,%.2f,%.2f,%.2f,%.3f,%.4f,%.6f,%.4f,%.1f\n",
                                arch_name, result.batch_size,
                                result.p9ml.memory.total,
                                result.p9ml.memory.forward,
                                result.p9ml.memory.backward,
                                result.p9ml.training.total_training_time,
                                result.p9ml.training.avg_batch_time,
                                result.p9ml.training.final_loss,
                                result.p9ml.inference.avg_inference_time,
                                result.p9ml.inference.throughput))
    end
    
    file:close()
    print(string.format("Results exported to %s", filename))
end

return P9MLPerformanceBenchmark