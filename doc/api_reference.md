# P9ML API Reference

> **Complete API Reference for P9ML Membrane Computing System Components**

## Table of Contents
- [P9MLMembrane API](#p9mlmembrane-api)
- [P9MLNamespace API](#p9mlnamespace-api)
- [P9MLCognitiveKernel API](#p9mlcognitivekernel-api)
- [P9MLEvolution API](#p9mlevolution-api)
- [P9MLVisualizer API](#p9mlvisualizer-api)
- [P9MLTest API](#p9mltest-api)
- [Factory Methods](#factory-methods)

---

## P9MLMembrane API

### Constructor

#### `nn.P9MLMembrane(module, membrane_id)`
Creates a new P9ML membrane wrapper around a neural network module.

**Parameters:**
- `module` (nn.Module): The neural network module to wrap
- `membrane_id` (string, optional): Unique identifier for the membrane

**Returns:** P9MLMembrane instance

**Example:**
```lua
local linear = nn.Linear(784, 128)
local membrane = nn.P9MLMembrane(linear, 'feature_extractor')
```

### Methods

#### `membrane:addEvolutionRule(rule)`
Adds an evolution rule to the membrane.

**Parameters:**
- `rule` (P9MLEvolutionRule): Evolution rule instance

**Returns:** void

**Example:**
```lua
local rule = nn.P9MLEvolutionFactory.createGradientEvolution(0.01, 0.9)
membrane:addEvolutionRule(rule)
```

#### `membrane:enableQuantization(bits, scale_factor)`
Enables quantization aware training for the membrane.

**Parameters:**
- `bits` (number): Target quantization bits (typically 8, 16)
- `scale_factor` (number): Initial scale factor for quantization

**Returns:** void

**Example:**
```lua
membrane:enableQuantization(8, 0.1)
```

#### `membrane:updateOutput(input)`
Performs forward pass through the membrane with cognitive enhancements.

**Parameters:**
- `input` (Tensor): Input tensor

**Returns:** Tensor - Enhanced output

#### `membrane:updateGradInput(input, gradOutput)`
Performs backward pass through the membrane with evolution.

**Parameters:**
- `input` (Tensor): Input tensor from forward pass
- `gradOutput` (Tensor): Gradient with respect to output

**Returns:** Tensor - Gradient with respect to input

#### `membrane:getMembraneInfo()`
Returns comprehensive membrane information.

**Returns:** Table with fields:
- `membrane_id` (string): Membrane identifier
- `tensor_vocabulary` (table): Analyzed tensor shapes
- `evolution_rules` (table): Applied evolution rules
- `qat_state` (table): Quantization state
- `wrapped_module` (string): Type of wrapped module

**Example:**
```lua
local info = membrane:getMembraneInfo()
print(string.format("Membrane %s has %d vocabulary entries", 
                   info.membrane_id, #info.tensor_vocabulary))
```

---

## P9MLNamespace API

### Constructor

#### `nn.P9MLNamespace(namespace_id)`
Creates a new P9ML namespace for distributed coordination.

**Parameters:**
- `namespace_id` (string, optional): Unique identifier for the namespace

**Returns:** P9MLNamespace instance

### Methods

#### `namespace:registerMembrane(membrane, registration_key)`
Registers a membrane for distributed computation.

**Parameters:**
- `membrane` (P9MLMembrane): Membrane to register
- `registration_key` (string, optional): Custom registration key

**Returns:** string - Registration key used

**Example:**
```lua
local key = namespace:registerMembrane(membrane, 'input_processor')
```

#### `namespace:orchestrateComputation(input, computation_graph)`
Orchestrates computation across registered membranes.

**Parameters:**
- `input` (Tensor): Input data
- `computation_graph` (table): Computation graph specification

**Returns:** Table - Computation results

#### `namespace:applyMetaLearning()`
Applies meta-learning across the namespace.

**Returns:** void

#### `namespace:getNamespaceState()`
Returns current namespace state and statistics.

**Returns:** Table with fields:
- `registered_count` (number): Number of registered membranes
- `hypergraph_stats` (table): Hypergraph topology statistics
- `meta_learning_stats` (table): Meta-learning performance

**Example:**
```lua
local state = namespace:getNamespaceState()
print(string.format("Namespace has %d membranes and %d hypergraph nodes",
                   state.registered_count, state.hypergraph_stats.nodes))
```

---

## P9MLCognitiveKernel API

### Constructor

#### `nn.P9MLCognitiveKernel()`
Creates a new cognitive kernel for hypergraph-based reasoning.

**Returns:** P9MLCognitiveKernel instance

### Methods

#### `kernel:addLexeme(tensor_shape, membrane_id, context)`
Adds a tensor shape as a lexeme to the cognitive lexicon.

**Parameters:**
- `tensor_shape` (table): Tensor shape as array (e.g., {784, 128})
- `membrane_id` (string): Associated membrane identifier
- `context` (table, optional): Contextual information

**Returns:** string - Lexeme identifier

**Example:**
```lua
local lexeme_id = kernel:addLexeme({784, 128}, 'feature_layer', {
    layer_type = 'linear',
    position = 'input'
})
```

#### `kernel:addGrammarRule(membrane_info, rule_type)`
Adds a membrane as a grammar rule in the production system.

**Parameters:**
- `membrane_info` (table): Membrane information from `getMembraneInfo()`
- `rule_type` (string): Type of grammar rule

**Returns:** string - Grammar rule identifier

#### `kernel:generateGestaltField()`
Generates a unified gestalt tensor field from all components.

**Returns:** Tensor - Gestalt field representation

#### `kernel:resolveFrameProblem(context, query_tensor)`
Resolves the frame problem using nested membrane embeddings.

**Parameters:**
- `context` (table): Problem context
- `query_tensor` (Tensor): Query tensor for resolution

**Returns:** Table with fields:
- `nested_contexts` (table): Resolved nested contexts
- `cognitive_coherence` (number): Coherence measure
- `resolution_path` (table): Solution path

#### `kernel:getCognitiveState()`
Returns current cognitive state and statistics.

**Returns:** Table with fields:
- `lexemes_count` (number): Number of lexemes
- `grammar_rules_count` (number): Number of grammar rules
- `gestalt_coherence` (number): Current gestalt coherence
- `production_categories` (table): Production rule categories

---

## P9MLEvolution API

### Constructor

#### `nn.P9MLEvolutionRule(rule_type, parameters)`
Creates a new evolution rule.

**Parameters:**
- `rule_type` (string): Type of evolution rule
- `parameters` (table): Rule parameters

**Returns:** P9MLEvolutionRule instance

### Methods

#### `rule:apply(membrane_objects)`
Applies the evolution rule to membrane objects.

**Parameters:**
- `membrane_objects` (table): Table of membrane objects

**Returns:** void

#### `rule:getEvolutionStats()`
Returns evolution rule statistics.

**Returns:** Table with fields:
- `rule_type` (string): Rule type
- `activation_count` (number): Number of activations
- `success_rate` (number): Success rate (0-1)
- `adaptation_history` (table): Adaptation history

---

## P9MLVisualizer API

### Static Methods

#### `P9MLVisualizer.generateFullSystemDiagram(namespace, kernel)`
Generates a complete system visualization.

**Parameters:**
- `namespace` (P9MLNamespace): Namespace instance
- `kernel` (P9MLCognitiveKernel): Cognitive kernel instance

**Returns:** string - ASCII diagram

#### `P9MLVisualizer.generateGraphvizDot(namespace, kernel)`
Generates Graphviz DOT format for system visualization.

**Parameters:**
- `namespace` (P9MLNamespace): Namespace instance  
- `kernel` (P9MLCognitiveKernel): Cognitive kernel instance

**Returns:** string - DOT format string

#### `P9MLVisualizer.generateVisualizationSuite(namespace, kernel, output_dir)`
Generates complete visualization suite.

**Parameters:**
- `namespace` (P9MLNamespace): Namespace instance
- `kernel` (P9MLCognitiveKernel): Cognitive kernel instance
- `output_dir` (string): Output directory path

**Returns:** Table - Paths to generated files

---

## P9MLTest API

### Static Methods

#### `P9MLTest.runAllTests()`
Runs the complete P9ML test suite.

**Returns:** boolean - True if all tests pass

#### `P9MLTest.testMembraneCreation()`
Tests membrane creation and basic functionality.

**Returns:** boolean - Test result

#### `P9MLTest.testNamespaceOperations()`
Tests namespace operations and coordination.

**Returns:** boolean - Test result

#### `P9MLTest.testCognitiveKernel()`
Tests cognitive kernel functionality.

**Returns:** boolean - Test result

#### `P9MLTest.testEvolutionRules()`
Tests evolution rule application and adaptation.

**Returns:** boolean - Test result

---

## Factory Methods

### P9MLEvolutionFactory

#### `nn.P9MLEvolutionFactory.createGradientEvolution(learning_rate, momentum)`
Creates a gradient-based evolution rule.

**Parameters:**
- `learning_rate` (number): Learning rate parameter
- `momentum` (number): Momentum parameter

**Returns:** P9MLEvolutionRule instance

**Example:**
```lua
local rule = nn.P9MLEvolutionFactory.createGradientEvolution(0.01, 0.9)
```

#### `nn.P9MLEvolutionFactory.createAdaptiveQuantization(target_bits, scale_factor)`
Creates an adaptive quantization rule.

**Parameters:**
- `target_bits` (number): Target quantization bits
- `scale_factor` (number): Initial scale factor

**Returns:** P9MLEvolutionRule instance

#### `nn.P9MLEvolutionFactory.createCognitiveAdaptation(threshold, strength)`
Creates a cognitive adaptation rule.

**Parameters:**
- `threshold` (number): Cognitive threshold
- `strength` (number): Adaptation strength

**Returns:** P9MLEvolutionRule instance

---

## Usage Patterns

### Basic Integration Pattern
```lua
-- 1. Create membrane-wrapped layers
local membrane1 = nn.P9MLMembrane(nn.Linear(784, 128), 'input_layer')
local membrane2 = nn.P9MLMembrane(nn.Linear(128, 10), 'output_layer')

-- 2. Add evolution rules
membrane1:addEvolutionRule(nn.P9MLEvolutionFactory.createGradientEvolution(0.01, 0.9))
membrane2:enableQuantization(8, 0.1)

-- 3. Create cognitive infrastructure
local namespace = nn.P9MLNamespace('mnist_classifier')
local kernel = nn.P9MLCognitiveKernel()

-- 4. Register and configure
namespace:registerMembrane(membrane1, 'feature_extractor')
namespace:registerMembrane(membrane2, 'classifier')
kernel:addLexeme({784, 128}, 'feature_transformation')
kernel:addLexeme({128, 10}, 'classification')
```

### Advanced Cognitive Pattern
```lua
-- Create complex cognitive network
local namespace = nn.P9MLNamespace('cognitive_system')
local kernel = nn.P9MLCognitiveKernel()

-- Build network with multiple membranes
local membranes = {}
for i = 1, 5 do
    local membrane = nn.P9MLMembrane(nn.Linear(128, 128), 'layer_' .. i)
    membrane:addEvolutionRule(nn.P9MLEvolutionFactory.createCognitiveAdaptation(0.5, 0.1))
    namespace:registerMembrane(membrane)
    kernel:addLexeme({128, 128}, 'layer_' .. i)
    table.insert(membranes, membrane)
end

-- Apply meta-learning and generate visualizations
namespace:applyMetaLearning()
local gestalt = kernel:generateGestaltField()
local visualizations = nn.P9MLVisualizer.generateVisualizationSuite(namespace, kernel, './viz')
```

---

## Error Handling

Most P9ML methods include error checking and will throw descriptive error messages for:
- Invalid tensor shapes
- Incompatible module types
- Missing required parameters
- Namespace registration conflicts
- Evolution rule application failures

Always wrap P9ML operations in appropriate error handling:

```lua
local success, err = pcall(function()
    local membrane = nn.P9MLMembrane(invalid_module, 'test')
end)

if not success then
    print("P9ML Error: " .. err)
end
```

---

## Performance Considerations

- **Memory Usage**: P9ML adds cognitive state overhead (~10-20% of base model)
- **Computation**: Evolution rules add minimal overhead (~1-5% per forward pass)
- **Quantization**: Can reduce memory usage by 50-75% with minimal accuracy loss
- **Cognitive Operations**: Gestalt field generation is O(n²) in number of lexemes

---

## See Also

- [📖 Main README](../README.md)
- [🏗️ Technical Architecture](../ARCHITECTURE.md)
- [🧠 P9ML Integration Guide](p9ml_integration.md)
- [🧪 Examples](../examples/p9ml_example.lua)