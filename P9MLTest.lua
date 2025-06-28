-- P9ML Membrane Computing System Tests
-- Comprehensive test suite for P9ML integration with Neural Network Package

require('torch')

local P9MLTest = {}

-- Test utilities
function P9MLTest.assertEqual(actual, expected, message)
   if actual ~= expected then
      error(string.format("Assertion failed: %s. Expected %s, got %s", 
                         message or "values not equal", tostring(expected), tostring(actual)))
   end
end

function P9MLTest.assertTensorEqual(actual, expected, tolerance, message)
   tolerance = tolerance or 1e-6
   if torch.isTensor(actual) and torch.isTensor(expected) then
      local diff = torch.abs(actual - expected):max()
      if diff > tolerance then
         error(string.format("Tensor assertion failed: %s. Max difference: %f", 
                            message or "tensors not equal", diff))
      end
   else
      error("assertTensorEqual requires tensor inputs")
   end
end

function P9MLTest.assertNotNil(value, message)
   if value == nil then
      error(message or "Expected non-nil value")
   end
end

function P9MLTest.assertTrue(condition, message)
   if not condition then
      error(message or "Expected true condition")
   end
end

-- Test P9ML Membrane functionality
function P9MLTest.testMembraneCreation()
   print("Testing P9ML Membrane creation...")
   
   -- Create base neural network module
   local linear = nn.Linear(10, 5)
   
   -- Wrap in P9ML membrane
   local membrane = nn.P9MLMembrane(linear, 'test_membrane_001')
   
   -- Test basic properties
   P9MLTest.assertNotNil(membrane, "Membrane should be created")
   P9MLTest.assertEqual(membrane.membrane_id, 'test_membrane_001', "Membrane ID should match")
   P9MLTest.assertNotNil(membrane.tensor_vocabulary, "Tensor vocabulary should exist")
   P9MLTest.assertNotNil(membrane.membrane_objects, "Membrane objects should exist")
   
   -- Test tensor vocabulary analysis
   P9MLTest.assertTrue(#membrane.tensor_vocabulary > 0, "Should have tensor vocabulary entries")
   
   for i, vocab_entry in pairs(membrane.tensor_vocabulary) do
      P9MLTest.assertNotNil(vocab_entry.shape, "Vocabulary entry should have shape")
      P9MLTest.assertNotNil(vocab_entry.complexity, "Vocabulary entry should have complexity")
      P9MLTest.assertNotNil(vocab_entry.param_type, "Vocabulary entry should have parameter type")
   end
   
   print("✓ Membrane creation test passed")
end

function P9MLTest.testMembraneForwardPass()
   print("Testing P9ML Membrane forward pass...")
   
   local linear = nn.Linear(10, 5)
   local membrane = nn.P9MLMembrane(linear)
   
   local input = torch.randn(3, 10)
   local output = membrane:forward(input)
   
   P9MLTest.assertNotNil(output, "Forward pass should produce output")
   P9MLTest.assertEqual(output:size(1), 3, "Batch dimension should be preserved")
   P9MLTest.assertEqual(output:size(2), 5, "Output dimension should match Linear layer")
   
   -- Test that membrane transformation is applied
   local linear_output = linear:forward(input)
   local diff = torch.abs(output - linear_output):max()
   P9MLTest.assertTrue(diff > 0, "Membrane should apply transformation (outputs should differ)")
   
   print("✓ Membrane forward pass test passed")
end

function P9MLTest.testMembraneQuantization()
   print("Testing P9ML Membrane quantization...")
   
   local linear = nn.Linear(10, 5)
   local membrane = nn.P9MLMembrane(linear)
   
   -- Enable quantization
   membrane:enableQuantization(8, 0.1)
   P9MLTest.assertTrue(membrane.qat_state.quantized, "Quantization should be enabled")
   P9MLTest.assertEqual(membrane.qat_state.precision_bits, 8, "Precision bits should be set")
   
   local input = torch.randn(3, 10)
   local output = membrane:forward(input)
   
   P9MLTest.assertNotNil(output, "Quantized forward pass should work")
   
   print("✓ Membrane quantization test passed")
end

function P9MLTest.testMembraneEvolution()
   print("Testing P9ML Membrane evolution...")
   
   local linear = nn.Linear(10, 5)
   local membrane = nn.P9MLMembrane(linear)
   
   -- Add evolution rules
   local grad_evolution = nn.P9MLEvolutionFactory.createGradientEvolution(0.01, 0.9)
   local weight_decay = nn.P9MLEvolutionFactory.createWeightDecay(0.0001)
   
   membrane:addEvolutionRule(grad_evolution)
   membrane:addEvolutionRule(weight_decay)
   
   P9MLTest.assertEqual(#membrane.evolution_rules, 2, "Should have 2 evolution rules")
   
   -- Test evolution application
   local input = torch.randn(3, 10)
   local target = torch.randn(3, 5)
   local criterion = nn.MSECriterion()
   
   local output = membrane:forward(input)
   local loss = criterion:forward(output, target)
   local grad_output = criterion:backward(output, target)
   membrane:backward(input, grad_output)
   
   -- Check that membrane objects have evolved
   for i, membrane_obj in ipairs(membrane.membrane_objects) do
      P9MLTest.assertNotNil(membrane_obj.evolution_state, "Membrane object should have evolution state")
   end
   
   print("✓ Membrane evolution test passed")
end

-- Test P9ML Namespace functionality
function P9MLTest.testNamespaceCreation()
   print("Testing P9ML Namespace creation...")
   
   local namespace = nn.P9MLNamespace('test_namespace')
   
   P9MLTest.assertNotNil(namespace, "Namespace should be created")
   P9MLTest.assertEqual(namespace.namespace_id, 'test_namespace', "Namespace ID should match")
   P9MLTest.assertNotNil(namespace.registered_membranes, "Should have membrane registry")
   P9MLTest.assertNotNil(namespace.hypergraph_topology, "Should have hypergraph topology")
   
   print("✓ Namespace creation test passed")
end

function P9MLTest.testNamespaceMembraneRegistration()
   print("Testing P9ML Namespace membrane registration...")
   
   local namespace = nn.P9MLNamespace('test_namespace')
   
   -- Create and register membranes
   local linear1 = nn.Linear(10, 5)
   local linear2 = nn.Linear(5, 3)
   local membrane1 = nn.P9MLMembrane(linear1, 'membrane_1')
   local membrane2 = nn.P9MLMembrane(linear2, 'membrane_2')
   
   local key1 = namespace:registerMembrane(membrane1)
   local key2 = namespace:registerMembrane(membrane2)
   
   P9MLTest.assertNotNil(key1, "Registration should return key")
   P9MLTest.assertNotNil(key2, "Registration should return key")
   P9MLTest.assertEqual(key1, 'membrane_1', "Key should match membrane ID")
   
   local state = namespace:getNamespaceState()
   P9MLTest.assertEqual(state.registered_count, 2, "Should have 2 registered membranes")
   
   print("✓ Namespace membrane registration test passed")
end

function P9MLTest.testNamespaceOrchestration()
   print("Testing P9ML Namespace orchestration...")
   
   local namespace = nn.P9MLNamespace('test_namespace')
   
   -- Create sequential network with membranes
   local linear1 = nn.Linear(10, 8)
   local linear2 = nn.Linear(8, 5)
   local membrane1 = nn.P9MLMembrane(linear1, 'layer_1')
   local membrane2 = nn.P9MLMembrane(linear2, 'layer_2')
   
   namespace:registerMembrane(membrane1)
   namespace:registerMembrane(membrane2)
   
   -- Test computation orchestration
   local input = torch.randn(3, 10)
   local computation_graph = {}  -- Simple graph for now
   local results = namespace:orchestrateComputation(input, computation_graph)
   
   P9MLTest.assertNotNil(results, "Orchestration should return results")
   P9MLTest.assertTrue(type(results) == 'table', "Results should be a table")
   
   print("✓ Namespace orchestration test passed")
end

-- Test P9ML Cognitive Kernel functionality
function P9MLTest.testCognitiveKernelCreation()
   print("Testing P9ML Cognitive Kernel creation...")
   
   local kernel = nn.P9MLCognitiveKernel()
   
   P9MLTest.assertNotNil(kernel, "Cognitive kernel should be created")
   P9MLTest.assertNotNil(kernel.hypergraph, "Should have hypergraph structure")
   P9MLTest.assertNotNil(kernel.cognitive_field, "Should have cognitive field")
   P9MLTest.assertNotNil(kernel.productions, "Should have production system")
   
   print("✓ Cognitive kernel creation test passed")
end

function P9MLTest.testCognitiveLexemeManagement()
   print("Testing P9ML Cognitive Kernel lexeme management...")
   
   local kernel = nn.P9MLCognitiveKernel()
   
   -- Add lexemes with different tensor shapes
   local shape1 = {10, 5}     -- Linear weight matrix
   local shape2 = {5}         -- Bias vector
   local shape3 = {3, 3, 32, 64}  -- Convolution kernel
   
   local lexeme1 = kernel:addLexeme(shape1, 'membrane_1', {layer_type = 'linear'})
   local lexeme2 = kernel:addLexeme(shape2, 'membrane_1', {layer_type = 'bias'})
   local lexeme3 = kernel:addLexeme(shape3, 'membrane_2', {layer_type = 'convolution'})
   
   P9MLTest.assertNotNil(lexeme1, "Should create lexeme 1")
   P9MLTest.assertNotNil(lexeme2, "Should create lexeme 2")
   P9MLTest.assertNotNil(lexeme3, "Should create lexeme 3")
   
   local state = kernel:getCognitiveState()
   P9MLTest.assertEqual(state.lexemes_count, 3, "Should have 3 lexemes")
   
   -- Test prime factor extraction
   local lexeme_data = kernel.hypergraph.lexemes[lexeme1]
   P9MLTest.assertNotNil(lexeme_data.prime_factors, "Lexeme should have prime factors")
   P9MLTest.assertNotNil(lexeme_data.grammatical_role, "Lexeme should have grammatical role")
   
   print("✓ Cognitive lexeme management test passed")
end

function P9MLTest.testCognitiveGrammarRules()
   print("Testing P9ML Cognitive Kernel grammar rules...")
   
   local kernel = nn.P9MLCognitiveKernel()
   
   -- Create membrane and add as grammar rule
   local linear = nn.Linear(10, 5)
   local membrane = nn.P9MLMembrane(linear, 'test_membrane')
   local membrane_info = membrane:getMembraneInfo()
   
   local rule_id = kernel:addGrammarRule(membrane_info, 'transformation')
   
   P9MLTest.assertNotNil(rule_id, "Should create grammar rule")
   
   local rule_data = kernel.hypergraph.grammar_rules[rule_id]
   P9MLTest.assertNotNil(rule_data, "Grammar rule should exist")
   P9MLTest.assertNotNil(rule_data.productions, "Rule should have productions")
   P9MLTest.assertTrue(#rule_data.productions > 0, "Should have production rules")
   
   local state = kernel:getCognitiveState()
   P9MLTest.assertEqual(state.grammar_rules_count, 1, "Should have 1 grammar rule")
   
   print("✓ Cognitive grammar rules test passed")
end

function P9MLTest.testCognitiveGestaltField()
   print("Testing P9ML Cognitive Kernel gestalt field...")
   
   local kernel = nn.P9MLCognitiveKernel()
   
   -- Add multiple lexemes and grammar rules
   kernel:addLexeme({10, 5}, 'membrane_1')
   kernel:addLexeme({5}, 'membrane_1')
   kernel:addLexeme({20, 10}, 'membrane_2')
   
   local linear = nn.Linear(10, 5)
   local membrane = nn.P9MLMembrane(linear)
   kernel:addGrammarRule(membrane:getMembraneInfo())
   
   -- Generate gestalt field
   local gestalt_tensor = kernel:generateGestaltField()
   
   P9MLTest.assertNotNil(gestalt_tensor, "Should generate gestalt tensor")
   P9MLTest.assertTrue(torch.isTensor(gestalt_tensor), "Gestalt should be tensor")
   P9MLTest.assertTrue(gestalt_tensor:nElement() > 0, "Gestalt tensor should have elements")
   
   local state = kernel:getCognitiveState()
   P9MLTest.assertTrue(state.gestalt_coherence >= 0, "Coherence should be non-negative")
   
   print("✓ Cognitive gestalt field test passed")
end

function P9MLTest.testFrameProblemResolution()
   print("Testing P9ML frame problem resolution...")
   
   local kernel = nn.P9MLCognitiveKernel()
   
   local context = {
      task = 'classification',
      layer_type = 'linear',
      input_size = 10
   }
   local query_tensor = torch.randn(5, 10)
   
   local resolution = kernel:resolveFrameProblem(context, query_tensor)
   
   P9MLTest.assertNotNil(resolution, "Should generate resolution")
   P9MLTest.assertNotNil(resolution.primary_context, "Should have primary context")
   P9MLTest.assertNotNil(resolution.nested_contexts, "Should have nested contexts")
   P9MLTest.assertNotNil(resolution.cognitive_coherence, "Should have coherence measure")
   
   print("✓ Frame problem resolution test passed")
end

-- Test P9ML Evolution Rules
function P9MLTest.testEvolutionRuleCreation()
   print("Testing P9ML Evolution Rule creation...")
   
   local grad_rule = nn.P9MLEvolutionFactory.createGradientEvolution(0.01, 0.9)
   local decay_rule = nn.P9MLEvolutionFactory.createWeightDecay(0.0001)
   local quantum_rule = nn.P9MLEvolutionFactory.createQuantumFluctuation(0.001)
   
   P9MLTest.assertNotNil(grad_rule, "Should create gradient evolution rule")
   P9MLTest.assertNotNil(decay_rule, "Should create weight decay rule")
   P9MLTest.assertNotNil(quantum_rule, "Should create quantum fluctuation rule")
   
   P9MLTest.assertEqual(grad_rule.rule_type, 'gradient_evolution', "Rule type should match")
   P9MLTest.assertEqual(decay_rule.rule_type, 'weight_decay', "Rule type should match")
   P9MLTest.assertEqual(quantum_rule.rule_type, 'quantum_fluctuation', "Rule type should match")
   
   print("✓ Evolution rule creation test passed")
end

function P9MLTest.testEvolutionRuleApplication()
   print("Testing P9ML Evolution Rule application...")
   
   local linear = nn.Linear(10, 5)
   local membrane = nn.P9MLMembrane(linear)
   
   -- Get initial weights
   local initial_weights = linear.weight:clone()
   
   -- Create and apply evolution rule
   local grad_rule = nn.P9MLEvolutionFactory.createGradientEvolution(0.1, 0.9)
   
   -- Simulate gradient computation
   local input = torch.randn(3, 10)
   local target = torch.randn(3, 5)
   local criterion = nn.MSECriterion()
   
   local output = membrane:forward(input)
   local loss = criterion:forward(output, target)
   local grad_output = criterion:backward(output, target)
   membrane:backward(input, grad_output)
   
   -- Apply evolution rule
   grad_rule:apply(membrane.membrane_objects)
   
   P9MLTest.assertTrue(grad_rule.activation_count > 0, "Rule should have been activated")
   
   -- Check that weights have evolved
   local weight_diff = torch.abs(linear.weight - initial_weights):max()
   P9MLTest.assertTrue(weight_diff > 0, "Weights should have evolved")
   
   print("✓ Evolution rule application test passed")
end

-- Integration tests
function P9MLTest.testFullP9MLIntegration()
   print("Testing full P9ML integration...")
   
   -- Create namespace and cognitive kernel
   local namespace = nn.P9MLNamespace('integration_test')
   local kernel = nn.P9MLCognitiveKernel()
   
   -- Create neural network with P9ML membranes
   local net = nn.Sequential()
   local linear1 = nn.Linear(20, 10)
   local linear2 = nn.Linear(10, 5)
   local linear3 = nn.Linear(5, 2)
   
   local membrane1 = nn.P9MLMembrane(linear1, 'layer1')
   local membrane2 = nn.P9MLMembrane(linear2, 'layer2')
   local membrane3 = nn.P9MLMembrane(linear3, 'layer3')
   
   -- Add evolution rules
   membrane1:addEvolutionRule(nn.P9MLEvolutionFactory.createGradientEvolution())
   membrane2:addEvolutionRule(nn.P9MLEvolutionFactory.createWeightDecay())
   membrane3:addEvolutionRule(nn.P9MLEvolutionFactory.createAdaptiveQuantization())
   
   net:add(membrane1)
   net:add(nn.ReLU())
   net:add(membrane2)
   net:add(nn.ReLU())
   net:add(membrane3)
   
   -- Register membranes in namespace
   namespace:registerMembrane(membrane1)
   namespace:registerMembrane(membrane2)
   namespace:registerMembrane(membrane3)
   
   -- Add to cognitive kernel
   kernel:addLexeme({20, 10}, 'layer1')
   kernel:addLexeme({10, 5}, 'layer2')
   kernel:addLexeme({5, 2}, 'layer3')
   
   kernel:addGrammarRule(membrane1:getMembraneInfo())
   kernel:addGrammarRule(membrane2:getMembraneInfo())
   kernel:addGrammarRule(membrane3:getMembraneInfo())
   
   -- Test forward pass
   local input = torch.randn(4, 20)
   local output = net:forward(input)
   
   P9MLTest.assertNotNil(output, "Network should produce output")
   P9MLTest.assertEqual(output:size(1), 4, "Batch size should be preserved")
   P9MLTest.assertEqual(output:size(2), 2, "Output size should be correct")
   
   -- Test backward pass
   local target = torch.randn(4, 2)
   local criterion = nn.MSECriterion()
   local loss = criterion:forward(output, target)
   local grad_output = criterion:backward(output, target)
   net:backward(input, grad_output)
   
   P9MLTest.assertTrue(loss >= 0, "Loss should be non-negative")
   
   -- Generate gestalt field
   local gestalt = kernel:generateGestaltField()
   P9MLTest.assertNotNil(gestalt, "Should generate gestalt field")
   
   -- Test namespace state
   local ns_state = namespace:getNamespaceState()
   P9MLTest.assertEqual(ns_state.registered_count, 3, "Should have 3 registered membranes")
   
   -- Test cognitive state
   local cog_state = kernel:getCognitiveState()
   P9MLTest.assertEqual(cog_state.lexemes_count, 3, "Should have 3 lexemes")
   P9MLTest.assertEqual(cog_state.grammar_rules_count, 3, "Should have 3 grammar rules")
   
   print("✓ Full P9ML integration test passed")
end

function P9MLTest.testMetaLearningLoop()
   print("Testing P9ML meta-learning loop...")
   
   local namespace = nn.P9MLNamespace('meta_test')
   
   -- Create membranes
   local membrane1 = nn.P9MLMembrane(nn.Linear(10, 5), 'meta_layer1')
   local membrane2 = nn.P9MLMembrane(nn.Linear(5, 3), 'meta_layer2')
   
   namespace:registerMembrane(membrane1)
   namespace:registerMembrane(membrane2)
   
   -- Add meta-rule (simple adaptation rule)
   local meta_rule = {
      apply = function(self, namespace)
         -- Simple meta-learning: adjust evolution rules based on performance
         for key, membrane in pairs(namespace.registered_membranes) do
            local registry = namespace.membrane_registry[key]
            if registry.activity_level > 10 then
               -- High activity: add more evolution rules
               if #membrane.evolution_rules < 3 then
                  membrane:addEvolutionRule(nn.P9MLEvolutionFactory.createCognitiveAdaptation())
               end
            end
         end
      end
   }
   
   namespace:addMetaRule(meta_rule)
   
   -- Simulate activity
   local input = torch.randn(2, 10)
   for i = 1, 15 do
      membrane1:forward(input)
      membrane2:forward(torch.randn(2, 5))
   end
   
   -- Apply meta-learning
   namespace:applyMetaLearning()
   
   -- Check that meta-learning had effect
   P9MLTest.assertTrue(#membrane1.evolution_rules > 0, "Meta-learning should add evolution rules")
   
   print("✓ Meta-learning loop test passed")
end

-- Test runner
function P9MLTest.runAllTests()
   print("="*60)
   print("Running P9ML Membrane Computing System Tests")
   print("="*60)
   
   local tests = {
      P9MLTest.testMembraneCreation,
      P9MLTest.testMembraneForwardPass,
      P9MLTest.testMembraneQuantization,
      P9MLTest.testMembraneEvolution,
      P9MLTest.testNamespaceCreation,
      P9MLTest.testNamespaceMembraneRegistration,
      P9MLTest.testNamespaceOrchestration,
      P9MLTest.testCognitiveKernelCreation,
      P9MLTest.testCognitiveLexemeManagement,
      P9MLTest.testCognitiveGrammarRules,
      P9MLTest.testCognitiveGestaltField,
      P9MLTest.testFrameProblemResolution,
      P9MLTest.testEvolutionRuleCreation,
      P9MLTest.testEvolutionRuleApplication,
      P9MLTest.testFullP9MLIntegration,
      P9MLTest.testMetaLearningLoop
   }
   
   local passed = 0
   local failed = 0
   
   for i, test in ipairs(tests) do
      local success, error_msg = pcall(test)
      if success then
         passed = passed + 1
      else
         failed = failed + 1
         print("✗ Test failed: " .. error_msg)
      end
   end
   
   print("\n" .. "="*60)
   print(string.format("Test Results: %d passed, %d failed", passed, failed))
   print("="*60)
   
   if failed == 0 then
      print("🎉 All P9ML tests passed! The membrane computing integration is working correctly.")
   else
      print("⚠️  Some tests failed. Please review the implementation.")
   end
   
   return failed == 0
end

-- Export test module
return P9MLTest