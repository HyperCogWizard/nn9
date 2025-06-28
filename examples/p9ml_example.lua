#!/usr/bin/env lua
-- P9ML Membrane Computing System Example
-- Demonstrates the basic usage of P9ML integration with neural networks

require('torch')
require('nn')

print("="*60)
print("P9ML Membrane Computing System - Integration Example")
print("="*60)

-- Create a simple neural network
print("\n1. Creating base neural network...")
local net = nn.Sequential()
local linear1 = nn.Linear(10, 8)
local linear2 = nn.Linear(8, 5)  
local linear3 = nn.Linear(5, 2)

net:add(linear1)
net:add(nn.ReLU())
net:add(linear2)
net:add(nn.ReLU())
net:add(linear3)
net:add(nn.Sigmoid())

print("✓ Base network created with 3 linear layers")

-- Wrap layers in P9ML membranes
print("\n2. Wrapping layers in P9ML membranes...")
local membrane1 = nn.P9MLMembrane(linear1, 'input_layer')
local membrane2 = nn.P9MLMembrane(linear2, 'hidden_layer')  
local membrane3 = nn.P9MLMembrane(linear3, 'output_layer')

-- Replace layers with membranes
net = nn.Sequential()
net:add(membrane1)
net:add(nn.ReLU())
net:add(membrane2)
net:add(nn.ReLU())
net:add(membrane3)
net:add(nn.Sigmoid())

print("✓ Layers wrapped in P9ML membranes")

-- Add evolution rules
print("\n3. Adding evolution rules to membranes...")
membrane1:addEvolutionRule(nn.P9MLEvolutionFactory.createGradientEvolution(0.01, 0.9))
membrane2:addEvolutionRule(nn.P9MLEvolutionFactory.createAdaptiveQuantization(8, 0.1))
membrane3:addEvolutionRule(nn.P9MLEvolutionFactory.createCognitiveAdaptation(0.01, 0.9))

-- Enable quantization on some membranes
membrane2:enableQuantization(8, 0.1)

print("✓ Evolution rules added")
print("  - Gradient evolution on input layer")
print("  - Adaptive quantization on hidden layer") 
print("  - Cognitive adaptation on output layer")

-- Create P9ML namespace for distributed management
print("\n4. Creating P9ML namespace...")
local namespace = nn.P9MLNamespace('example_network')

namespace:registerMembrane(membrane1, 'input')
namespace:registerMembrane(membrane2, 'hidden')
namespace:registerMembrane(membrane3, 'output')

print("✓ Namespace created and membranes registered")

-- Create cognitive kernel
print("\n5. Creating cognitive kernel...")
local kernel = nn.P9MLCognitiveKernel()

-- Add lexemes (tensor shapes as vocabulary)
kernel:addLexeme({10, 8}, 'input', {layer_type = 'linear', position = 'input'})
kernel:addLexeme({8, 5}, 'hidden', {layer_type = 'linear', position = 'hidden'})
kernel:addLexeme({5, 2}, 'output', {layer_type = 'linear', position = 'output'})

-- Add grammar rules (membranes as transformations)
kernel:addGrammarRule(membrane1:getMembraneInfo(), 'input_transformation')
kernel:addGrammarRule(membrane2:getMembraneInfo(), 'feature_extraction')
kernel:addGrammarRule(membrane3:getMembraneInfo(), 'output_transformation')

print("✓ Cognitive kernel created with lexemes and grammar rules")

-- Demonstrate forward pass
print("\n6. Testing forward pass...")
local input = torch.randn(4, 10)  -- Batch of 4 samples
local output = net:forward(input)

print(string.format("✓ Forward pass successful"))
print(string.format("  Input shape: %s", table.concat(input:size():totable(), 'x')))
print(string.format("  Output shape: %s", table.concat(output:size():totable(), 'x')))

-- Demonstrate backward pass  
print("\n7. Testing backward pass...")
local target = torch.randn(4, 2)
local criterion = nn.MSECriterion()
local loss = criterion:forward(output, target)
local grad_output = criterion:backward(output, target)
local grad_input = net:backward(input, grad_output)

print(string.format("✓ Backward pass successful"))
print(string.format("  Loss: %.6f", loss))

-- Show membrane information
print("\n8. Membrane analysis...")
for i, membrane in ipairs({membrane1, membrane2, membrane3}) do
    local info = membrane:getMembraneInfo()
    print(string.format("  Membrane %d (%s):", i, info.membrane_id))
    print(string.format("    Vocabulary entries: %d", #info.tensor_vocabulary))
    print(string.format("    Evolution rules: %d", #info.evolution_rules))
    print(string.format("    Quantized: %s", info.qat_state.quantized and "Yes" or "No"))
end

-- Generate gestalt field
print("\n9. Generating cognitive gestalt field...")
local gestalt = kernel:generateGestaltField()
local cognitive_state = kernel:getCognitiveState()

print("✓ Gestalt field generated")
print(string.format("  Gestalt tensor size: %s", table.concat(gestalt:size():totable(), 'x')))
print(string.format("  Cognitive coherence: %.4f", cognitive_state.gestalt_coherence))
print(string.format("  Lexemes: %d", cognitive_state.lexemes_count))
print(string.format("  Grammar rules: %d", cognitive_state.grammar_rules_count))

-- Demonstrate namespace orchestration
print("\n10. Testing namespace orchestration...")
local computation_graph = {}  -- Simple empty graph for this example
local results = namespace:orchestrateComputation(input, computation_graph)
local ns_state = namespace:getNamespaceState()

print("✓ Namespace orchestration completed")
print(string.format("  Registered membranes: %d", ns_state.registered_count))
print(string.format("  Hypergraph nodes: %d", ns_state.hypergraph_stats.nodes))
print(string.format("  Hypergraph edges: %d", ns_state.hypergraph_stats.edges))

-- Apply meta-learning
print("\n11. Applying meta-learning...")
namespace:applyMetaLearning()

-- Test frame problem resolution
print("\n12. Testing frame problem resolution...")
local context = {
    task = 'regression',
    layer_type = 'linear',
    input_size = 10,
    output_size = 2
}
local query_tensor = torch.randn(3, 10)
local resolution = kernel:resolveFrameProblem(context, query_tensor)

print("✓ Frame problem resolution completed")
print(string.format("  Nested contexts: %d", #resolution.nested_contexts))
print(string.format("  Cognitive coherence: %.4f", resolution.cognitive_coherence))

-- Training simulation with P9ML evolution
print("\n13. Simulating training with P9ML evolution...")
local optimizer = {
    learningRate = 0.01,
    momentum = 0.9
}

for epoch = 1, 5 do
    local epoch_loss = 0
    for batch = 1, 10 do
        -- Generate random training data
        local train_input = torch.randn(4, 10)
        local train_target = torch.randn(4, 2)
        
        -- Forward pass
        local train_output = net:forward(train_input)
        local batch_loss = criterion:forward(train_output, train_target)
        epoch_loss = epoch_loss + batch_loss
        
        -- Backward pass
        local grad_out = criterion:backward(train_output, train_target)
        net:backward(train_input, grad_out)
        
        -- Update parameters (simplified)
        local params, grad_params = net:getParameters()
        params:add(grad_params:mul(-optimizer.learningRate))
        
        -- Apply namespace meta-learning every few batches
        if batch % 5 == 0 then
            namespace:applyMetaLearning()
        end
    end
    
    -- Update gestalt field periodically
    if epoch % 2 == 0 then
        kernel:generateGestaltField()
        local state = kernel:getCognitiveState()
        print(string.format("  Epoch %d: Loss=%.6f, Coherence=%.4f", 
                           epoch, epoch_loss/10, state.gestalt_coherence))
    else
        print(string.format("  Epoch %d: Loss=%.6f", epoch, epoch_loss/10))
    end
end

print("✓ Training simulation completed with P9ML evolution")

-- Final analysis
print("\n14. Final P9ML system analysis...")

-- Check evolution rule statistics
print("  Evolution Rule Statistics:")
for i, membrane in ipairs({membrane1, membrane2, membrane3}) do
    for j, rule in ipairs(membrane.evolution_rules) do
        local stats = rule:getEvolutionStats()
        print(string.format("    Membrane %d, Rule %d (%s): %d activations, %.3f success rate",
                           i, j, stats.rule_type, stats.activation_count, stats.success_rate))
    end
end

-- Final cognitive state
local final_cognitive_state = kernel:getCognitiveState()
print("\n  Final Cognitive State:")
print(string.format("    Lexemes: %d", final_cognitive_state.lexemes_count))
print(string.format("    Grammar rules: %d", final_cognitive_state.grammar_rules_count))
print(string.format("    Gestalt coherence: %.4f", final_cognitive_state.gestalt_coherence))
print("    Production categories:")
for category, count in pairs(final_cognitive_state.production_categories) do
    print(string.format("      %s: %d", category, count))
end

-- Final namespace state
local final_ns_state = namespace:getNamespaceState()
print("\n  Final Namespace State:")
print(string.format("    Registered membranes: %d", final_ns_state.registered_count))
print(string.format("    Hypergraph topology: %d nodes, %d edges", 
                   final_ns_state.hypergraph_stats.nodes, final_ns_state.hypergraph_stats.edges))

print("\n" .. "="*60)
print("🎉 P9ML Membrane Computing System Integration Successful!")
print("The neural network now has:")
print("  ✓ Membrane-embedded layers with cognitive capabilities")
print("  ✓ Distributed namespace management")
print("  ✓ Cognitive grammar kernel with hypergraph representation")  
print("  ✓ Evolution rules for adaptive behavior")
print("  ✓ Quantization aware training")
print("  ✓ Meta-learning loops for recursive adaptation")
print("  ✓ Frame problem resolution through nested embeddings")
print("="*60)

-- Optional: Run comprehensive tests
print("\nOptional: Run comprehensive test suite? (y/n)")
-- Uncomment next lines to auto-run tests
-- local P9MLTest = require('nn.P9MLTest')  
-- P9MLTest.runAllTests()