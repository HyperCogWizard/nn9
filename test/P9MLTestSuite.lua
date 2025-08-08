-- P9ML Membrane Computing System Test Suite Integration
-- Integrates P9ML tests into the main nn testing framework

require('torch')
require('nn')

-- Load P9ML test module
local P9MLTest = require('P9MLTest')

-- P9ML Test Suite for nn.test integration
local P9MLTestSuite = {}

-- Integration wrapper functions for nn.test framework
function P9MLTestSuite.P9MLMembrane()
   local mytester = torch.Tester()
   
   -- Basic membrane functionality
   local success, error_msg = pcall(P9MLTest.testMembraneCreation)
   mytester:assert(success, "P9ML Membrane Creation: " .. (error_msg or ""))
   
   local success, error_msg = pcall(P9MLTest.testMembraneForwardPass)
   mytester:assert(success, "P9ML Membrane Forward Pass: " .. (error_msg or ""))
   
   local success, error_msg = pcall(P9MLTest.testMembraneQuantization)
   mytester:assert(success, "P9ML Membrane Quantization: " .. (error_msg or ""))
   
   local success, error_msg = pcall(P9MLTest.testMembraneEvolution)
   mytester:assert(success, "P9ML Membrane Evolution: " .. (error_msg or ""))
   
   -- Advanced membrane functionality
   local success, error_msg = pcall(P9MLTest.testMembraneQuantumStates)
   mytester:assert(success, "P9ML Membrane Quantum States: " .. (error_msg or ""))
   
   local success, error_msg = pcall(P9MLTest.testMembraneGradientTracking)
   mytester:assert(success, "P9ML Membrane Gradient Tracking: " .. (error_msg or ""))
   
   local success, error_msg = pcall(P9MLTest.testMembraneErrorHandling)
   mytester:assert(success, "P9ML Membrane Error Handling: " .. (error_msg or ""))
end

function P9MLTestSuite.P9MLNamespace()
   local mytester = torch.Tester()
   
   -- Basic namespace functionality
   local success, error_msg = pcall(P9MLTest.testNamespaceCreation)
   mytester:assert(success, "P9ML Namespace Creation: " .. (error_msg or ""))
   
   local success, error_msg = pcall(P9MLTest.testNamespaceMembraneRegistration)
   mytester:assert(success, "P9ML Namespace Registration: " .. (error_msg or ""))
   
   local success, error_msg = pcall(P9MLTest.testNamespaceOrchestration)
   mytester:assert(success, "P9ML Namespace Orchestration: " .. (error_msg or ""))
   
   -- Advanced namespace functionality
   local success, error_msg = pcall(P9MLTest.testNamespaceHypergraphEvolution)
   mytester:assert(success, "P9ML Namespace Hypergraph Evolution: " .. (error_msg or ""))
   
   local success, error_msg = pcall(P9MLTest.testNamespaceMembraneInteraction)
   mytester:assert(success, "P9ML Namespace Membrane Interaction: " .. (error_msg or ""))
end

function P9MLTestSuite.P9MLCognitiveKernel()
   local mytester = torch.Tester()
   
   -- Basic cognitive kernel functionality
   local success, error_msg = pcall(P9MLTest.testCognitiveKernelCreation)
   mytester:assert(success, "P9ML Cognitive Kernel Creation: " .. (error_msg or ""))
   
   local success, error_msg = pcall(P9MLTest.testCognitiveLexemeManagement)
   mytester:assert(success, "P9ML Cognitive Lexeme Management: " .. (error_msg or ""))
   
   local success, error_msg = pcall(P9MLTest.testCognitiveGrammarRules)
   mytester:assert(success, "P9ML Cognitive Grammar Rules: " .. (error_msg or ""))
   
   local success, error_msg = pcall(P9MLTest.testCognitiveGestaltField)
   mytester:assert(success, "P9ML Cognitive Gestalt Field: " .. (error_msg or ""))
   
   local success, error_msg = pcall(P9MLTest.testFrameProblemResolution)
   mytester:assert(success, "P9ML Frame Problem Resolution: " .. (error_msg or ""))
   
   -- Advanced cognitive kernel functionality
   local success, error_msg = pcall(P9MLTest.testCognitivePrimeFactorAnalysis)
   mytester:assert(success, "P9ML Cognitive Prime Factor Analysis: " .. (error_msg or ""))
   
   local success, error_msg = pcall(P9MLTest.testCognitiveSemanticSimilarity)
   mytester:assert(success, "P9ML Cognitive Semantic Similarity: " .. (error_msg or ""))
   
   local success, error_msg = pcall(P9MLTest.testCognitiveGrammarProductions)
   mytester:assert(success, "P9ML Cognitive Grammar Productions: " .. (error_msg or ""))
end

function P9MLTestSuite.P9MLEvolution()
   local mytester = torch.Tester()
   
   -- Basic evolution functionality
   local success, error_msg = pcall(P9MLTest.testEvolutionRuleCreation)
   mytester:assert(success, "P9ML Evolution Rule Creation: " .. (error_msg or ""))
   
   local success, error_msg = pcall(P9MLTest.testEvolutionRuleApplication)
   mytester:assert(success, "P9ML Evolution Rule Application: " .. (error_msg or ""))
   
   -- Advanced evolution functionality
   local success, error_msg = pcall(P9MLTest.testAllEvolutionRuleTypes)
   mytester:assert(success, "P9ML All Evolution Rule Types: " .. (error_msg or ""))
   
   local success, error_msg = pcall(P9MLTest.testEvolutionRuleAdaptation)
   mytester:assert(success, "P9ML Evolution Rule Adaptation: " .. (error_msg or ""))
end

function P9MLTestSuite.P9MLIntegration()
   local mytester = torch.Tester()
   
   -- Integration tests
   local success, error_msg = pcall(P9MLTest.testMembraneToMembraneSignaling)
   mytester:assert(success, "P9ML Membrane-to-Membrane Signaling: " .. (error_msg or ""))
   
   local success, error_msg = pcall(P9MLTest.testMembraneCollaborativeEvolution)
   mytester:assert(success, "P9ML Membrane Collaborative Evolution: " .. (error_msg or ""))
   
   local success, error_msg = pcall(P9MLTest.testFullP9MLIntegration)
   mytester:assert(success, "P9ML Full Integration: " .. (error_msg or ""))
   
   local success, error_msg = pcall(P9MLTest.testMetaLearningLoop)
   mytester:assert(success, "P9ML Meta-Learning Loop: " .. (error_msg or ""))
end

function P9MLTestSuite.P9MLStressTesting()
   local mytester = torch.Tester()
   
   -- Stress tests
   local success, error_msg = pcall(P9MLTest.testLargeNetworkStressTest)
   mytester:assert(success, "P9ML Large Network Stress Test: " .. (error_msg or ""))
   
   local success, error_msg = pcall(P9MLTest.testMemoryStressTest)
   mytester:assert(success, "P9ML Memory Stress Test: " .. (error_msg or ""))
end

-- Test coverage reporting function
function P9MLTestSuite.reportCoverage()
   print("\n" .. "="*80)
   print("P9ML MEMBRANE COMPUTING SYSTEM - TEST COVERAGE REPORT")
   print("="*80)
   
   local coverage_info = P9MLTest.calculateTestCoverage()
   
   print(string.format("Overall Test Coverage: %.1f%%", coverage_info.estimated_coverage))
   print(string.format("Function Coverage: %.1f%% (%d/%d functions tested)", 
                      coverage_info.function_coverage,
                      coverage_info.total_functions_tested,
                      coverage_info.total_functions))
   print(string.format("Edge Case Coverage: %.1f%% (%d/%d edge cases tested)",
                      coverage_info.edge_case_coverage,
                      coverage_info.total_edge_cases_tested,
                      coverage_info.total_edge_cases))
   
   P9MLTest.printCoverageBreakdown(coverage_info)
   
   if coverage_info.estimated_coverage >= 95.0 then
      print("\n🎉 P9ML test coverage target achieved (95%+)!")
      print("✅ All P9ML components have comprehensive test coverage")
      print("✅ Integration tests for membrane-to-membrane communication included")
      print("✅ Stress testing for large neural networks implemented")
   else
      print(string.format("\n⚠️  P9ML test coverage is %.1f%%, below 95%% target", coverage_info.estimated_coverage))
      print("Consider adding more edge case tests or covering additional functionality")
   end
   
   return coverage_info.estimated_coverage >= 95.0
end

-- Run P9ML performance benchmarks
function P9MLTestSuite.runBenchmarks()
   print("\n" .. "="*80)
   print("P9ML MEMBRANE COMPUTING SYSTEM - PERFORMANCE BENCHMARKS")
   print("="*80)
   
   P9MLTest.runPerformanceBenchmarks()
   
   print("\nBenchmarks completed. Review performance metrics for optimization opportunities.")
end

-- Comprehensive P9ML test suite runner
function P9MLTestSuite.runFullSuite()
   print("="*80)
   print("COMPREHENSIVE P9ML MEMBRANE COMPUTING SYSTEM TEST SUITE")
   print("="*80)
   
   local mytester = torch.Tester()
   
   -- Add all P9ML tests to the tester
   mytester:add(P9MLTestSuite, 'P9ML')
   
   -- Run comprehensive test suite
   local success = P9MLTest.runAllTests()
   
   -- Report coverage
   local coverage_target_met = P9MLTestSuite.reportCoverage()
   
   -- Run performance benchmarks
   P9MLTestSuite.runBenchmarks()
   
   print("\n" .. "="*80)
   print("FINAL RESULTS")
   print("="*80)
   
   if success and coverage_target_met then
      print("🎉 ALL P9ML TESTS PASSED WITH 95%+ COVERAGE!")
      print("✅ P9ML Membrane Computing System is fully tested and ready for production")
   elseif success then
      print("⚠️  All tests passed but coverage target not met")
   else
      print("❌ Some tests failed - review implementation before deployment")
   end
   
   return success and coverage_target_met
end

return P9MLTestSuite