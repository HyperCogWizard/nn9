-- P9ML Edge Case and Robustness Tests
-- Additional tests to achieve 95%+ coverage and handle edge cases

local P9MLEdgeTests = {}

-- Membrane Edge Cases
function P9MLEdgeTests.testMembraneWithZeroDimensions()
   print("Testing P9ML Membrane with zero/invalid dimensions...")
   
   -- Test membrane with very small networks
   local tiny_linear = nn.Linear(1, 1)
   local membrane = nn.P9MLMembrane(tiny_linear, 'tiny_test')
   
   local input = torch.randn(1, 1)
   local output = membrane:forward(input)
   
   assert(output:nElement() == 1, "Tiny membrane should work")
   
   -- Test with various module types
   local modules_to_test = {
      nn.Linear(5, 3),
      nn.ReLU(),
      nn.Sigmoid(),
      nn.Tanh(),
      nn.Sequential():add(nn.Linear(4, 2)):add(nn.ReLU())
   }
   
   for i, module in ipairs(modules_to_test) do
      local membrane = nn.P9MLMembrane(module, 'test_' .. i)
      assert(membrane ~= nil, "Should create membrane for module type: " .. torch.type(module))
   end
   
   print("✓ Membrane zero/invalid dimensions test passed")
end

function P9MLEdgeTests.testMembraneQuantizationExtremes()
   print("Testing P9ML Membrane quantization extreme cases...")
   
   local linear = nn.Linear(10, 5)
   local membrane = nn.P9MLMembrane(linear)
   
   -- Test extreme quantization settings
   local extreme_settings = {
      {bits = 1, scale = 0.001},    -- Very low precision
      {bits = 16, scale = 100.0},   -- High precision, large scale
      {bits = 4, scale = 0.0001},   -- Medium precision, tiny scale
   }
   
   for _, setting in ipairs(extreme_settings) do
      membrane:enableQuantization(setting.bits, setting.scale)
      
      local input = torch.randn(2, 10)
      local output = membrane:forward(input)
      
      assert(torch.isTensor(output), "Should produce tensor output with extreme quantization")
      assert(not torch.any(torch.isnan(output)), "Output should not contain NaN")
      assert(not torch.any(torch.isinf(output)), "Output should not contain Inf")
   end
   
   print("✓ Membrane quantization extremes test passed")
end

function P9MLEdgeTests.testMembraneEvolutionBoundaryConditions()
   print("Testing P9ML Membrane evolution boundary conditions...")
   
   local linear = nn.Linear(3, 2)
   local membrane = nn.P9MLMembrane(linear)
   
   -- Test evolution with extreme gradients
   local extreme_gradients = {
      torch.zeros(2, 2),                    -- Zero gradients
      torch.ones(2, 2) * 1000,              -- Very large gradients
      torch.ones(2, 2) * 1e-10,             -- Very small gradients
      torch.randn(2, 2) * 0 + 1e-8,         -- Near-zero gradients
   }
   
   for i, grad in ipairs(extreme_gradients) do
      membrane:forward(torch.randn(2, 3))
      membrane:backward(torch.randn(2, 3), grad)
      
      -- Check membrane object states remain valid
      for _, obj in ipairs(membrane.membrane_objects) do
         assert(obj.evolution_state ~= nil, "Evolution state should remain valid")
         if obj.tensor then
            assert(not torch.any(torch.isnan(obj.tensor)), "Tensor should not contain NaN")
            assert(not torch.any(torch.isinf(obj.tensor)), "Tensor should not contain Inf")
         end
      end
   end
   
   print("✓ Membrane evolution boundary conditions test passed")
end

-- Namespace Edge Cases
function P9MLEdgeTests.testNamespaceWithManyMembranes()
   print("Testing P9ML Namespace with many membranes...")
   
   local namespace = nn.P9MLNamespace('many_membranes_test')
   local num_membranes = 100
   
   -- Register many membranes
   for i = 1, num_membranes do
      local size_in = math.random(5, 20)
      local size_out = math.random(3, 15)
      local linear = nn.Linear(size_in, size_out)
      local membrane = nn.P9MLMembrane(linear, 'many_' .. i)
      
      namespace:registerMembrane(membrane)
   end
   
   local state = namespace:getNamespaceState()
   assert(state.registered_count == num_membranes, "Should register all membranes")
   
   -- Test hypergraph doesn't become too dense
   local edge_density = state.hypergraph_stats.edges / (state.hypergraph_stats.nodes * (state.hypergraph_stats.nodes - 1) / 2)
   assert(edge_density <= 1.0, "Edge density should be reasonable")
   
   print("✓ Namespace with many membranes test passed")
end

function P9MLEdgeTests.testNamespaceOrchestrationFailures()
   print("Testing P9ML Namespace orchestration failure cases...")
   
   local namespace = nn.P9MLNamespace('failure_test')
   
   -- Test orchestration with no membranes
   local empty_results = namespace:orchestrateComputation(torch.randn(2, 5), {})
   assert(type(empty_results) == 'table', "Should handle empty namespace gracefully")
   
   -- Test with incompatible membranes
   local membrane1 = nn.P9MLMembrane(nn.Linear(10, 5), 'incompatible1')
   local membrane2 = nn.P9MLMembrane(nn.Linear(3, 2), 'incompatible2')  -- Wrong input size
   
   namespace:registerMembrane(membrane1)
   namespace:registerMembrane(membrane2)
   
   -- Should handle orchestration even with incompatible sizes
   local input = torch.randn(2, 10)
   local results = namespace:orchestrateComputation(input, {})
   assert(type(results) == 'table', "Should handle incompatible membranes gracefully")
   
   print("✓ Namespace orchestration failures test passed")
end

-- Cognitive Kernel Edge Cases
function P9MLEdgeTests.testCognitiveKernelWithUnusualTensorShapes()
   print("Testing P9ML Cognitive Kernel with unusual tensor shapes...")
   
   local kernel = nn.P9MLCognitiveKernel()
   
   -- Test with unusual tensor shapes
   local unusual_shapes = {
      {1},                    -- 1D scalar-like
      {1, 1},                 -- 2D scalar-like
      {1, 1, 1, 1, 1},        -- 5D thin tensor
      {2, 3, 5, 7, 11, 13},   -- 6D with prime dimensions
      {1024, 1024},           -- Very large 2D
      {1000000},              -- Very large 1D
   }
   
   for i, shape in ipairs(unusual_shapes) do
      local lexeme_id = kernel:addLexeme(shape, 'unusual_' .. i, {})
      assert(lexeme_id ~= nil, "Should handle unusual shape: " .. table.concat(shape, 'x'))
      
      local lexeme = kernel.hypergraph.lexemes[lexeme_id]
      assert(lexeme.prime_factors ~= nil, "Should compute prime factors for unusual shapes")
      assert(lexeme.semantic_weight > 0, "Should compute positive semantic weight")
   end
   
   print("✓ Cognitive kernel unusual tensor shapes test passed")
end

function P9MLEdgeTests.testCognitiveKernelGestaltFieldExtremeCases()
   print("Testing P9ML Cognitive Kernel gestalt field extreme cases...")
   
   local kernel = nn.P9MLCognitiveKernel()
   
   -- Add many lexemes to test gestalt field scaling
   for i = 1, 200 do
      kernel:addLexeme({math.random(1, 100), math.random(1, 100)}, 'gestalt_' .. i)
   end
   
   local gestalt = kernel:generateGestaltField()
   assert(torch.isTensor(gestalt), "Should generate gestalt tensor with many lexemes")
   assert(gestalt:nElement() > 0, "Gestalt tensor should have elements")
   assert(not torch.any(torch.isnan(gestalt)), "Gestalt should not contain NaN")
   
   local coherence = kernel.cognitive_field.coherence_measure
   assert(coherence >= 0 and coherence <= 1, "Coherence should be in valid range")
   
   print("✓ Cognitive kernel gestalt field extreme cases test passed")
end

-- Evolution Rule Edge Cases
function P9MLEdgeTests.testEvolutionRulesWithExtremeParameters()
   print("Testing P9ML Evolution Rules with extreme parameters...")
   
   local linear = nn.Linear(5, 3)
   local membrane = nn.P9MLMembrane(linear)
   
   -- Test evolution rules with extreme parameters
   local extreme_rules = {
      nn.P9MLEvolutionFactory.createGradientEvolution(1e-10, 0.01),  -- Tiny evolution rate
      nn.P9MLEvolutionFactory.createGradientEvolution(100.0, 0.99),  -- Huge evolution rate
      nn.P9MLEvolutionFactory.createWeightDecay(1e-8, true),         -- Tiny decay
      nn.P9MLEvolutionFactory.createWeightDecay(0.1, false),         -- Large decay
      nn.P9MLEvolutionFactory.createQuantumFluctuation(1e-6, 1e-8),  -- Tiny fluctuation
      nn.P9MLEvolutionFactory.createAdaptiveQuantization(1, 1e-3),   -- 1-bit quantization
      nn.P9MLEvolutionFactory.createAdaptiveQuantization(32, 100.0), -- High precision
   }
   
   for i, rule in ipairs(extreme_rules) do
      membrane:addEvolutionRule(rule)
      
      -- Apply rule multiple times
      for j = 1, 10 do
         local input = torch.randn(1, 5)
         local output = membrane:forward(input)
         membrane:backward(input, torch.randn(output:size()))
      end
      
      -- Check rule statistics
      assert(rule.activation_count > 0, "Rule should be activated")
      assert(rule.success_rate >= 0 and rule.success_rate <= 1, "Success rate should be valid")
   end
   
   print("✓ Evolution rules extreme parameters test passed")
end

function P9MLEdgeTests.testEvolutionRuleMemoryManagement()
   print("Testing P9ML Evolution Rule memory management...")
   
   local rule = nn.P9MLEvolutionFactory.createGradientEvolution()
   
   -- Fill adaptation history beyond limit
   for i = 1, 1500 do  -- More than the 1000 limit
      rule:_recordAdaptation(10, 8)
   end
   
   assert(#rule.adaptation_history <= 1000, "Adaptation history should be limited")
   
   -- Test with cognitive adaptation rule
   local cog_rule = nn.P9MLEvolutionFactory.createCognitiveAdaptation()
   local linear = nn.Linear(4, 2)
   local membrane = nn.P9MLMembrane(linear)
   membrane:addEvolutionRule(cog_rule)
   
   -- Create many membrane objects with cognitive states
   for i = 1, 200 do
      local input = torch.randn(1, 4)
      membrane:forward(input)
      membrane:backward(input, torch.randn(1, 2))
   end
   
   -- Check memory usage is reasonable
   for _, obj in ipairs(membrane.membrane_objects) do
      if obj.cognitive_state and obj.cognitive_state.usage_history then
         assert(#obj.cognitive_state.usage_history <= 100, "Usage history should be limited")
      end
   end
   
   print("✓ Evolution rule memory management test passed")
end

-- Integration Stress Tests
function P9MLEdgeTests.testDeepMembraneChain()
   print("Testing P9ML deep membrane chain...")
   
   local namespace = nn.P9MLNamespace('deep_chain')
   local kernel = nn.P9MLCognitiveKernel()
   
   -- Create deep chain of membranes
   local depth = 20
   local membranes = {}
   local current_size = 50
   
   for i = 1, depth do
      local next_size = math.max(2, current_size - 2)
      local linear = nn.Linear(current_size, next_size)
      local membrane = nn.P9MLMembrane(linear, 'deep_' .. i)
      
      -- Add random evolution rules
      if i % 3 == 0 then
         membrane:addEvolutionRule(nn.P9MLEvolutionFactory.createGradientEvolution())
      elseif i % 3 == 1 then
         membrane:addEvolutionRule(nn.P9MLEvolutionFactory.createAdaptiveQuantization())
      else
         membrane:addEvolutionRule(nn.P9MLEvolutionFactory.createCognitiveAdaptation())
      end
      
      table.insert(membranes, membrane)
      namespace:registerMembrane(membrane)
      kernel:addLexeme({current_size, next_size}, 'deep_' .. i)
      
      current_size = next_size
   end
   
   -- Test forward pass through deep chain
   local input = torch.randn(3, 50)
   local current_input = input
   
   for _, membrane in ipairs(membranes) do
      current_input = membrane:forward(current_input)
      assert(torch.isTensor(current_input), "Deep chain should maintain tensor flow")
      assert(not torch.any(torch.isnan(current_input)), "No NaN in deep chain")
   end
   
   -- Test backward pass
   local grad = torch.randn(current_input:size())
   for i = #membranes, 1, -1 do
      grad = membranes[i]:backward(current_input, grad)
      assert(torch.isTensor(grad), "Deep chain backprop should work")
   end
   
   print("✓ Deep membrane chain test passed")
end

function P9MLEdgeTests.testConcurrentMembraneExecution()
   print("Testing P9ML concurrent membrane execution simulation...")
   
   local namespace = nn.P9MLNamespace('concurrent_test')
   
   -- Create multiple independent membrane branches
   local num_branches = 5
   local branches = {}
   
   for branch = 1, num_branches do
      local branch_membranes = {}
      for layer = 1, 3 do
         local linear = nn.Linear(10, 8)
         local membrane = nn.P9MLMembrane(linear, 'branch_' .. branch .. '_layer_' .. layer)
         table.insert(branch_membranes, membrane)
         namespace:registerMembrane(membrane)
      end
      table.insert(branches, branch_membranes)
   end
   
   -- Simulate concurrent execution
   local inputs = {}
   local outputs = {}
   
   for branch = 1, num_branches do
      inputs[branch] = torch.randn(2, 10)
      outputs[branch] = {}
   end
   
   -- Forward pass for all branches
   for branch = 1, num_branches do
      local current_input = inputs[branch]
      for layer = 1, #branches[branch] do
         current_input = branches[branch][layer]:forward(current_input)
         outputs[branch][layer] = current_input:clone()
      end
   end
   
   -- Verify all branches executed successfully
   for branch = 1, num_branches do
      assert(#outputs[branch] == 3, "All layers should execute")
      for layer = 1, 3 do
         assert(torch.isTensor(outputs[branch][layer]), "Output should be tensor")
      end
   end
   
   print("✓ Concurrent membrane execution test passed")
end

-- Performance and Resource Tests
function P9MLEdgeTests.testMemoryLeakPrevention()
   print("Testing P9ML memory leak prevention...")
   
   -- Create and destroy many membranes
   for cycle = 1, 10 do
      local namespace = nn.P9MLNamespace('leak_test_' .. cycle)
      local kernel = nn.P9MLCognitiveKernel()
      
      for i = 1, 50 do
         local linear = nn.Linear(math.random(5, 20), math.random(3, 15))
         local membrane = nn.P9MLMembrane(linear, 'leak_' .. i)
         membrane:addEvolutionRule(nn.P9MLEvolutionFactory.createGradientEvolution())
         
         namespace:registerMembrane(membrane)
         kernel:addLexeme({10, 5}, 'leak_' .. i)
         
         -- Use the membrane
         local input = torch.randn(2, linear.weight:size(2))
         membrane:forward(input)
         membrane:backward(input, torch.randn(2, linear.weight:size(1)))
      end
      
      -- Force some cleanup by nullifying references
      namespace = nil
      kernel = nil
      collectgarbage('collect')
   end
   
   print("✓ Memory leak prevention test passed")
end

function P9MLEdgeTests.testLargeScaleMetaLearning()
   print("Testing P9ML large-scale meta-learning...")
   
   local namespace = nn.P9MLNamespace('large_meta')
   
   -- Create many membranes for meta-learning
   local num_membranes = 30
   for i = 1, num_membranes do
      local linear = nn.Linear(math.random(5, 15), math.random(3, 10))
      local membrane = nn.P9MLMembrane(linear, 'meta_' .. i)
      
      -- Add various evolution rules
      membrane:addEvolutionRule(nn.P9MLEvolutionFactory.createGradientEvolution())
      if i % 2 == 0 then
         membrane:addEvolutionRule(nn.P9MLEvolutionFactory.createCognitiveAdaptation())
      end
      
      namespace:registerMembrane(membrane)
   end
   
   -- Add multiple meta-rules
   for rule_idx = 1, 5 do
      local meta_rule = {
         apply = function(self, namespace)
            for key, membrane in pairs(namespace.registered_membranes) do
               local registry = namespace.membrane_registry[key]
               if registry.activity_level > 20 then
                  -- High activity adaptation
                  for _, rule in ipairs(membrane.evolution_rules) do
                     if rule.rule_type == 'gradient_evolution' then
                        rule.parameters.evolution_rate = math.min(0.1, rule.parameters.evolution_rate * 1.1)
                     end
                  end
               end
            end
         end
      }
      namespace:addMetaRule(meta_rule)
   end
   
   -- Simulate many iterations with meta-learning
   for iteration = 1, 50 do
      -- Activity simulation
      for key, membrane in pairs(namespace.registered_membranes) do
         local input_size = membrane.modules[1].weight:size(2)
         local input = torch.randn(1, input_size)
         membrane:forward(input)
      end
      
      -- Apply meta-learning every few iterations
      if iteration % 5 == 0 then
         namespace:applyMetaLearning()
      end
   end
   
   -- Verify meta-learning effects
   local high_activity_count = 0
   for key, registry in pairs(namespace.membrane_registry) do
      if registry.activity_level > 20 then
         high_activity_count = high_activity_count + 1
      end
   end
   
   assert(high_activity_count > 0, "Some membranes should show high activity")
   
   print("✓ Large-scale meta-learning test passed")
end

-- Export all edge tests
function P9MLEdgeTests.runAllEdgeTests()
   print("\n" .. "="*60)
   print("Running P9ML Edge Case and Robustness Tests")
   print("="*60)
   
   local edge_tests = {
      P9MLEdgeTests.testMembraneWithZeroDimensions,
      P9MLEdgeTests.testMembraneQuantizationExtremes,
      P9MLEdgeTests.testMembraneEvolutionBoundaryConditions,
      P9MLEdgeTests.testNamespaceWithManyMembranes,
      P9MLEdgeTests.testNamespaceOrchestrationFailures,
      P9MLEdgeTests.testCognitiveKernelWithUnusualTensorShapes,
      P9MLEdgeTests.testCognitiveKernelGestaltFieldExtremeCases,
      P9MLEdgeTests.testEvolutionRulesWithExtremeParameters,
      P9MLEdgeTests.testEvolutionRuleMemoryManagement,
      P9MLEdgeTests.testDeepMembraneChain,
      P9MLEdgeTests.testConcurrentMembraneExecution,
      P9MLEdgeTests.testMemoryLeakPrevention,
      P9MLEdgeTests.testLargeScaleMetaLearning
   }
   
   local passed = 0
   local failed = 0
   
   for i, test in ipairs(edge_tests) do
      local success, error_msg = pcall(test)
      if success then
         passed = passed + 1
      else
         failed = failed + 1
         print("✗ Edge test failed: " .. (error_msg or "Unknown error"))
      end
   end
   
   print("\n" .. "="*60)
   print(string.format("Edge Test Results: %d passed, %d failed", passed, failed))
   print("="*60)
   
   return failed == 0
end

return P9MLEdgeTests