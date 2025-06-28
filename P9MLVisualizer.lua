#!/usr/bin/env lua
-- P9ML Hypergraph Visualization Generator
-- Creates visual representations of membrane, namespace, and tensor relationships

require('torch')
require('nn')

local P9MLVisualizer = {}

function P9MLVisualizer.generateMembraneVisualization(membrane)
    -- Generate ASCII art visualization of membrane structure
    local info = membrane:getMembraneInfo()
    local viz_lines = {}
    
    table.insert(viz_lines, "┌" .. string.rep("─", 60) .. "┐")
    table.insert(viz_lines, string.format("│ P9ML Membrane: %-43s │", info.membrane_id))
    table.insert(viz_lines, "├" .. string.rep("─", 60) .. "┤")
    table.insert(viz_lines, string.format("│ Wrapped Module: %-42s │", info.wrapped_module))
    table.insert(viz_lines, "│                                                            │")
    
    -- Tensor vocabulary visualization
    table.insert(viz_lines, "│ Tensor Vocabulary:                                         │")
    for i, vocab_entry in pairs(info.tensor_vocabulary) do
        local shape_str = table.concat(vocab_entry.shape, "×")
        local complexity_str = string.format("%.2f", vocab_entry.complexity)
        table.insert(viz_lines, string.format("│  %d. %-15s [%-10s] complexity: %-10s │", 
                     i, vocab_entry.param_type, shape_str, complexity_str))
    end
    
    table.insert(viz_lines, "│                                                            │")
    
    -- Evolution rules
    table.insert(viz_lines, string.format("│ Evolution Rules: %-40s │", #info.evolution_rules))
    for i, rule in ipairs(membrane.evolution_rules) do
        local stats = rule:getEvolutionStats()
        table.insert(viz_lines, string.format("│  %d. %-20s (act: %d, succ: %.2f)            │", 
                     i, stats.rule_type, stats.activation_count, stats.success_rate))
    end
    
    table.insert(viz_lines, "│                                                            │")
    table.insert(viz_lines, string.format("│ Quantization: %-43s │", 
                info.qat_state.quantized and 
                string.format("Enabled (%d-bit)", info.qat_state.precision_bits) or "Disabled"))
    
    table.insert(viz_lines, "└" .. string.rep("─", 60) .. "┘")
    
    return table.concat(viz_lines, "\n")
end

function P9MLVisualizer.generateNamespaceTopology(namespace)
    -- Generate hypergraph topology visualization
    local state = namespace:getNamespaceState()
    local viz_lines = {}
    
    table.insert(viz_lines, "")
    table.insert(viz_lines, "P9ML Namespace Hypergraph Topology")
    table.insert(viz_lines, "=" .. string.rep("=", 35))
    table.insert(viz_lines, string.format("Namespace ID: %s", namespace.namespace_id))
    table.insert(viz_lines, string.format("Registered Membranes: %d", state.registered_count))
    table.insert(viz_lines, string.format("Hypergraph Nodes: %d", state.hypergraph_stats.nodes))
    table.insert(viz_lines, string.format("Hypergraph Edges: %d", state.hypergraph_stats.edges))
    table.insert(viz_lines, "")
    
    -- Node visualization
    table.insert(viz_lines, "Nodes (Membranes):")
    for key, node in pairs(namespace.hypergraph_topology.nodes) do
        local membrane = namespace.registered_membranes[key]
        local connections = #(node.connections or {})
        table.insert(viz_lines, string.format("  %s: %d connections", key, connections))
        
        -- Show connections
        if connections > 0 then
            for _, connected_key in ipairs(node.connections) do
                table.insert(viz_lines, string.format("    └─→ %s", connected_key))
            end
        end
    end
    
    table.insert(viz_lines, "")
    
    -- Edge visualization
    table.insert(viz_lines, "Edges (Cognitive Connections):")
    for edge_id, edge in pairs(namespace.hypergraph_topology.edges) do
        table.insert(viz_lines, string.format("  %s ↔ %s (strength: %.3f)", 
                     edge.source, edge.target, edge.strength))
    end
    
    return table.concat(viz_lines, "\n")
end

function P9MLVisualizer.generateCognitiveKernelMap(kernel)
    -- Generate cognitive kernel hypergraph map
    local state = kernel:getCognitiveState()
    local viz_lines = {}
    
    table.insert(viz_lines, "")
    table.insert(viz_lines, "P9ML Cognitive Kernel Hypergraph")
    table.insert(viz_lines, "=" .. string.rep("=", 33))
    table.insert(viz_lines, string.format("Lexemes: %d", state.lexemes_count))
    table.insert(viz_lines, string.format("Grammar Rules: %d", state.grammar_rules_count))
    table.insert(viz_lines, string.format("Gestalt Coherence: %.4f", state.gestalt_coherence))
    table.insert(viz_lines, "")
    
    -- Lexemes as nodes
    table.insert(viz_lines, "Lexemes (Tensor Shapes as Vocabulary):")
    for lexeme_id, lexeme in pairs(kernel.hypergraph.lexemes) do
        local shape_str = table.concat(lexeme.shape, "×")
        local role = lexeme.grammatical_role
        table.insert(viz_lines, string.format("  %s: [%s] (%s)", 
                     lexeme_id:sub(1, 12) .. "...", shape_str, role))
    end
    
    table.insert(viz_lines, "")
    
    -- Grammar rules
    table.insert(viz_lines, "Grammar Rules (Membranes as Transformations):")
    for rule_id, rule in pairs(kernel.hypergraph.grammar_rules) do
        table.insert(viz_lines, string.format("  %s: %s (strength: %.2f)", 
                     rule_id, rule.rule_type, rule.cognitive_strength))
        table.insert(viz_lines, string.format("    Productions: %d", #rule.productions))
    end
    
    table.insert(viz_lines, "")
    
    -- Production categories
    table.insert(viz_lines, "Production Categories:")
    for category, count in pairs(state.production_categories) do
        if count > 0 then
            table.insert(viz_lines, string.format("  %s: %d productions", category, count))
        end
    end
    
    -- Prime factor catalog
    table.insert(viz_lines, "")
    table.insert(viz_lines, "Prime Factor Catalog:")
    local factor_count = 0
    for factor, entries in pairs(kernel.cognitive_field.prime_factors) do
        if factor_count < 10 then  -- Limit display
            table.insert(viz_lines, string.format("  Prime %d: %d occurrences", factor, #entries))
            factor_count = factor_count + 1
        end
    end
    
    return table.concat(viz_lines, "\n")
end

function P9MLVisualizer.generateFullSystemDiagram(namespace, kernel)
    -- Generate comprehensive system diagram
    local viz_lines = {}
    
    table.insert(viz_lines, "")
    table.insert(viz_lines, "┌" .. string.rep("─", 78) .. "┐")
    table.insert(viz_lines, "│" .. string.rep(" ", 20) .. "P9ML MEMBRANE COMPUTING SYSTEM" .. string.rep(" ", 27) .. "│")
    table.insert(viz_lines, "├" .. string.rep("─", 78) .. "┤")
    table.insert(viz_lines, "│                                                                              │")
    
    -- Layer 1: Neural Network Layers → P9ML Membranes
    table.insert(viz_lines, "│ Layer 1: Neural Network Layers ↔ P9ML Membranes                            │")
    table.insert(viz_lines, "│   [Linear] → [P9MLMembrane] → Tensor Vocabulary + Evolution Rules          │")
    table.insert(viz_lines, "│   [Conv2D] → [P9MLMembrane] → Quantum States + QAT                         │")
    table.insert(viz_lines, "│                                                                              │")
    
    -- Layer 2: Namespace Management
    table.insert(viz_lines, "│ Layer 2: Distributed Namespace Management                                   │")
    table.insert(viz_lines, "│   P9MLNamespace: Membrane Registry + Hypergraph Topology                   │")
    table.insert(viz_lines, "│   Orchestration: Cognitive Similarity → Hypergraph Connections             │")
    table.insert(viz_lines, "│                                                                              │")
    
    -- Layer 3: Cognitive Grammar Kernel  
    table.insert(viz_lines, "│ Layer 3: Cognitive Grammar Kernel                                           │")
    table.insert(viz_lines, "│   Lexemes: Tensor Shapes → Prime Factor Decomposition                      │")
    table.insert(viz_lines, "│   Grammar Rules: Membranes → Production Systems                            │")
    table.insert(viz_lines, "│   Meta-Grammar: Namespaces → Meta-Learning Rules                           │")
    table.insert(viz_lines, "│                                                                              │")
    
    -- Layer 4: Gestalt Field
    table.insert(viz_lines, "│ Layer 4: Unified Gestalt Tensor Field                                       │")
    table.insert(viz_lines, "│   Frame Problem Resolution: Nested Membrane Embeddings                     │")
    table.insert(viz_lines, "│   Cognitive Coherence: Eigenvalue-based Field Coherence                    │")
    table.insert(viz_lines, "│                                                                              │")
    
    -- Statistics
    local ns_state = namespace:getNamespaceState()
    local cog_state = kernel:getCognitiveState()
    
    table.insert(viz_lines, "│ System Statistics:                                                           │")
    table.insert(viz_lines, string.format("│   Registered Membranes: %-10d  Lexemes: %-10d              │", 
                 ns_state.registered_count, cog_state.lexemes_count))
    table.insert(viz_lines, string.format("│   Hypergraph Edges: %-14d  Grammar Rules: %-10d          │", 
                 ns_state.hypergraph_stats.edges, cog_state.grammar_rules_count))
    table.insert(viz_lines, string.format("│   Cognitive Coherence: %-37.4f              │", 
                 cog_state.gestalt_coherence))
    table.insert(viz_lines, "│                                                                              │")
    table.insert(viz_lines, "└" .. string.rep("─", 78) .. "┘")
    
    return table.concat(viz_lines, "\n")
end

function P9MLVisualizer.generateGraphvizDot(namespace, kernel)
    -- Generate Graphviz DOT format for external visualization
    local dot_lines = {}
    
    table.insert(dot_lines, "digraph P9MLSystem {")
    table.insert(dot_lines, "  rankdir=TB;")
    table.insert(dot_lines, "  node [shape=box, style=rounded];")
    table.insert(dot_lines, "")
    
    -- Membrane nodes
    table.insert(dot_lines, "  // P9ML Membranes")
    for key, membrane in pairs(namespace.registered_membranes) do
        local info = membrane:getMembraneInfo()
        local vocab_count = 0
        for _ in pairs(info.tensor_vocabulary) do vocab_count = vocab_count + 1 end
        
        table.insert(dot_lines, string.format('  "%s" [label="%s\\nVocab: %d\\nRules: %d", fillcolor=lightblue, style=filled];', 
                     key, info.membrane_id, vocab_count, #info.evolution_rules))
    end
    
    table.insert(dot_lines, "")
    
    -- Namespace node
    table.insert(dot_lines, "  // Namespace")
    table.insert(dot_lines, string.format('  "namespace" [label="Namespace\\n%s\\nMembranes: %d", fillcolor=lightgreen, style=filled];', 
                 namespace.namespace_id, namespace:getNamespaceState().registered_count))
    
    table.insert(dot_lines, "")
    
    -- Cognitive kernel node
    table.insert(dot_lines, "  // Cognitive Kernel")
    local cog_state = kernel:getCognitiveState()
    table.insert(dot_lines, string.format('  "kernel" [label="Cognitive Kernel\\nLexemes: %d\\nRules: %d\\nCoherence: %.3f", fillcolor=lightyellow, style=filled];', 
                 cog_state.lexemes_count, cog_state.grammar_rules_count, cog_state.gestalt_coherence))
    
    table.insert(dot_lines, "")
    
    -- Connections
    table.insert(dot_lines, "  // Connections")
    for key, _ in pairs(namespace.registered_membranes) do
        table.insert(dot_lines, string.format('  "%s" -> "namespace";', key))
        table.insert(dot_lines, string.format('  "%s" -> "kernel";', key))
    end
    table.insert(dot_lines, '  "namespace" -> "kernel";')
    
    -- Hypergraph edges
    for edge_id, edge in pairs(namespace.hypergraph_topology.edges) do
        table.insert(dot_lines, string.format('  "%s" -> "%s" [label="%.2f", color=red];', 
                     edge.source, edge.target, edge.strength))
    end
    
    table.insert(dot_lines, "}")
    
    return table.concat(dot_lines, "\n")
end

function P9MLVisualizer.saveVisualization(filename, content)
    -- Save visualization to file
    local file = io.open(filename, "w")
    if file then
        file:write(content)
        file:close()
        return true
    else
        return false
    end
end

function P9MLVisualizer.generateAllVisualizations(namespace, kernel, membranes, output_dir)
    -- Generate all visualization types
    output_dir = output_dir or "/tmp/p9ml_visualizations"
    
    -- Create output directory
    os.execute("mkdir -p " .. output_dir)
    
    local visualizations = {}
    
    -- Individual membrane visualizations
    for i, membrane in ipairs(membranes or {}) do
        local membrane_viz = P9MLVisualizer.generateMembraneVisualization(membrane)
        local filename = string.format("%s/membrane_%d.txt", output_dir, i)
        P9MLVisualizer.saveVisualization(filename, membrane_viz)
        visualizations[string.format("membrane_%d", i)] = filename
    end
    
    -- Namespace topology
    local namespace_viz = P9MLVisualizer.generateNamespaceTopology(namespace)
    local namespace_file = output_dir .. "/namespace_topology.txt"
    P9MLVisualizer.saveVisualization(namespace_file, namespace_viz)
    visualizations.namespace = namespace_file
    
    -- Cognitive kernel map
    local kernel_viz = P9MLVisualizer.generateCognitiveKernelMap(kernel)
    local kernel_file = output_dir .. "/cognitive_kernel.txt"
    P9MLVisualizer.saveVisualization(kernel_file, kernel_viz)
    visualizations.kernel = kernel_file
    
    -- Full system diagram
    local system_viz = P9MLVisualizer.generateFullSystemDiagram(namespace, kernel)
    local system_file = output_dir .. "/system_diagram.txt"
    P9MLVisualizer.saveVisualization(system_file, system_viz)
    visualizations.system = system_file
    
    -- Graphviz DOT file
    local dot_viz = P9MLVisualizer.generateGraphvizDot(namespace, kernel)
    local dot_file = output_dir .. "/p9ml_system.dot"
    P9MLVisualizer.saveVisualization(dot_file, dot_viz)
    visualizations.graphviz = dot_file
    
    return visualizations
end

-- Example usage function
function P9MLVisualizer.demo()
    -- Create demo P9ML system
    print("Creating demo P9ML system for visualization...")
    
    local namespace = nn.P9MLNamespace('demo_system')
    local kernel = nn.P9MLCognitiveKernel()
    
    -- Create membranes
    local linear1 = nn.Linear(10, 8)
    local linear2 = nn.Linear(8, 5)
    
    local membrane1 = nn.P9MLMembrane(linear1, 'demo_layer1')
    local membrane2 = nn.P9MLMembrane(linear2, 'demo_layer2')
    
    -- Add evolution rules
    membrane1:addEvolutionRule(nn.P9MLEvolutionFactory.createGradientEvolution())
    membrane2:addEvolutionRule(nn.P9MLEvolutionFactory.createAdaptiveQuantization())
    
    -- Register in namespace
    namespace:registerMembrane(membrane1)
    namespace:registerMembrane(membrane2)
    
    -- Add to cognitive kernel
    kernel:addLexeme({10, 8}, 'demo_layer1')
    kernel:addLexeme({8, 5}, 'demo_layer2')
    kernel:addGrammarRule(membrane1:getMembraneInfo())
    kernel:addGrammarRule(membrane2:getMembraneInfo())
    
    -- Generate visualizations
    print("\nGenerating visualizations...")
    local viz_files = P9MLVisualizer.generateAllVisualizations(
        namespace, kernel, {membrane1, membrane2}
    )
    
    print("\nVisualization files created:")
    for name, filename in pairs(viz_files) do
        print(string.format("  %s: %s", name, filename))
    end
    
    -- Print system diagram to console
    print("\n" .. P9MLVisualizer.generateFullSystemDiagram(namespace, kernel))
    
    return viz_files
end

return P9MLVisualizer