-- P9ML Cognitive Grammar Kernel
-- Hypergraph-based cognitive lexicon and grammar transformation system
local P9MLCognitiveKernel = torch.class('nn.P9MLCognitiveKernel')

function P9MLCognitiveKernel:__init()
   -- Core cognitive structures
   self.hypergraph = {
      lexemes = {},           -- Tensor shapes as lexemes
      grammar_rules = {},     -- Membranes as grammar rules  
      meta_grammar = {},      -- Namespaces as meta-grammar
      field_tensors = {},     -- Gestalt tensor field
      transformation_rules = {}
   }
   
   -- Cognitive field properties
   self.cognitive_field = {
      prime_factors = {},
      unique_shapes = {},
      gestalt_state = torch.Tensor(),
      coherence_measure = 0.0
   }
   
   -- Grammar production system
   self.productions = {
      lexical = {},
      syntactic = {},
      semantic = {},
      pragmatic = {}
   }
   
   -- Frame problem resolution structures
   self.frame_embeddings = {}
   self.nested_contexts = {}
end

function P9MLCognitiveKernel:addLexeme(tensor_shape, membrane_id, context)
   -- Add tensor shape as lexeme to cognitive lexicon
   local lexeme_id = self:_generateLexemeId(tensor_shape)
   local prime_factors = self:_factorizeTensorShape(tensor_shape)
   
   self.hypergraph.lexemes[lexeme_id] = {
      shape = tensor_shape,
      prime_factors = prime_factors,
      membrane_id = membrane_id,
      context = context or {},
      usage_frequency = 1,
      semantic_weight = self:_calculateSemanticWeight(tensor_shape),
      grammatical_role = self:_inferGrammaticalRole(tensor_shape, context)
   }
   
   -- Update prime factor catalog
   self:_updatePrimeFactorCatalog(prime_factors, lexeme_id)
   
   -- Create hypergraph connections
   self:_createLexicalConnections(lexeme_id)
   
   return lexeme_id
end

function P9MLCognitiveKernel:_generateLexemeId(tensor_shape)
   -- Generate unique lexeme identifier from tensor shape
   local shape_str = table.concat(tensor_shape, 'x')
   local hash = 0
   for i = 1, #shape_str do
      hash = ((hash * 37) + string.byte(shape_str, i)) % 1000000
   end
   return 'lexeme_' .. hash
end

function P9MLCognitiveKernel:_factorizeTensorShape(tensor_shape)
   -- Extract prime factors from tensor dimensions
   local prime_factors = {}
   
   for i, dim in ipairs(tensor_shape) do
      local factors = self:_primeFactorization(dim)
      prime_factors[i] = factors
   end
   
   return prime_factors
end

function P9MLCognitiveKernel:_primeFactorization(n)
   -- Simple prime factorization
   local factors = {}
   local d = 2
   
   while d * d <= n do
      while n % d == 0 do
         table.insert(factors, d)
         n = n / d
      end
      d = d + 1
   end
   
   if n > 1 then
      table.insert(factors, n)
   end
   
   return factors
end

function P9MLCognitiveKernel:_calculateSemanticWeight(tensor_shape)
   -- Calculate semantic weight based on tensor complexity
   local total_elements = 1
   local dimensionality = #tensor_shape
   
   for _, dim in ipairs(tensor_shape) do
      total_elements = total_elements * dim
   end
   
   -- Logarithmic scaling for semantic weight
   return math.log(total_elements) + dimensionality * 0.5
end

function P9MLCognitiveKernel:_inferGrammaticalRole(tensor_shape, context)
   -- Infer grammatical role from tensor shape and context
   local dimensionality = #tensor_shape
   
   if dimensionality == 1 then
      return 'vector_noun'  -- Bias vectors, embeddings
   elseif dimensionality == 2 then
      if tensor_shape[1] == tensor_shape[2] then
         return 'matrix_verb'  -- Square transformation matrices
      else
         return 'matrix_adjective'  -- Rectangular weight matrices
      end
   elseif dimensionality == 3 then
      return 'tensor_adverb'  -- Convolution kernels, attention heads
   elseif dimensionality >= 4 then
      return 'hypercube_interjection'  -- Complex multi-dimensional tensors
   else
      return 'scalar_article'  -- Scalar parameters
   end
end

function P9MLCognitiveKernel:_updatePrimeFactorCatalog(prime_factors, lexeme_id)
   -- Update catalog of prime factors for gestalt field
   for dim_idx, factors in ipairs(prime_factors) do
      for _, factor in ipairs(factors) do
         if not self.cognitive_field.prime_factors[factor] then
            self.cognitive_field.prime_factors[factor] = {}
         end
         
         table.insert(self.cognitive_field.prime_factors[factor], {
            lexeme_id = lexeme_id,
            dimension = dim_idx,
            power = self:_countOccurrences(factors, factor)
         })
      end
   end
end

function P9MLCognitiveKernel:_countOccurrences(table, element)
   local count = 0
   for _, value in ipairs(table) do
      if value == element then
         count = count + 1
      end
   end
   return count
end

function P9MLCognitiveKernel:_createLexicalConnections(lexeme_id)
   -- Create hypergraph connections between lexemes
   local current_lexeme = self.hypergraph.lexemes[lexeme_id]
   
   for other_lexeme_id, other_lexeme in pairs(self.hypergraph.lexemes) do
      if other_lexeme_id ~= lexeme_id then
         local similarity = self:_calculateLexicalSimilarity(current_lexeme, other_lexeme)
         
         if similarity > 0.3 then  -- Similarity threshold
            self:_createHyperedge(lexeme_id, other_lexeme_id, similarity, 'lexical')
         end
      end
   end
end

function P9MLCognitiveKernel:_calculateLexicalSimilarity(lexeme1, lexeme2)
   -- Calculate similarity between two lexemes
   local shape_similarity = self:_compareShapes(lexeme1.shape, lexeme2.shape)
   local semantic_similarity = 1.0 / (1.0 + math.abs(lexeme1.semantic_weight - lexeme2.semantic_weight))
   local role_similarity = (lexeme1.grammatical_role == lexeme2.grammatical_role) and 1.0 or 0.0
   
   return (shape_similarity + semantic_similarity + role_similarity) / 3.0
end

function P9MLCognitiveKernel:_compareShapes(shape1, shape2)
   -- Compare tensor shapes for similarity
   if #shape1 ~= #shape2 then
      return 0.0
   end
   
   local similarity = 0.0
   for i = 1, #shape1 do
      local ratio = math.min(shape1[i], shape2[i]) / math.max(shape1[i], shape2[i])
      similarity = similarity + ratio
   end
   
   return similarity / #shape1
end

function P9MLCognitiveKernel:_createHyperedge(node1, node2, weight, edge_type)
   -- Create hyperedge in cognitive graph
   local edge_id = node1 .. '_' .. node2 .. '_' .. edge_type
   
   if not self.hypergraph.transformation_rules[edge_type] then
      self.hypergraph.transformation_rules[edge_type] = {}
   end
   
   self.hypergraph.transformation_rules[edge_type][edge_id] = {
      source = node1,
      target = node2,
      weight = weight,
      activation_count = 0,
      transformation_type = edge_type,
      cognitive_resonance = weight
   }
end

function P9MLCognitiveKernel:addGrammarRule(membrane_info, rule_type)
   -- Add membrane as grammar rule
   local rule_id = 'rule_' .. membrane_info.membrane_id
   
   self.hypergraph.grammar_rules[rule_id] = {
      membrane_id = membrane_info.membrane_id,
      tensor_vocabulary = membrane_info.tensor_vocabulary,
      rule_type = rule_type or 'transformation',
      productions = {},
      activation_pattern = {},
      cognitive_strength = self:_calculateCognitiveStrength(membrane_info)
   }
   
   -- Generate grammar productions from membrane
   self:_generateGrammarProductions(rule_id, membrane_info)
   
   return rule_id
end

function P9MLCognitiveKernel:_calculateCognitiveStrength(membrane_info)
   -- Calculate cognitive strength of membrane as grammar rule
   local vocabulary_size = 0
   local total_complexity = 0
   
   for _, vocab_entry in pairs(membrane_info.tensor_vocabulary) do
      vocabulary_size = vocabulary_size + 1
      total_complexity = total_complexity + vocab_entry.complexity
   end
   
   local evolution_factor = #membrane_info.evolution_rules > 0 and 1.5 or 1.0
   local quantization_factor = membrane_info.qat_state.quantized and 1.2 or 1.0
   
   return (total_complexity / math.max(vocabulary_size, 1)) * evolution_factor * quantization_factor
end

function P9MLCognitiveKernel:_generateGrammarProductions(rule_id, membrane_info)
   -- Generate grammar productions from membrane information
   local productions = {}
   
   for vocab_id, vocab_entry in pairs(membrane_info.tensor_vocabulary) do
      local production = {
         input_shape = vocab_entry.shape,
         output_transformation = self:_inferTransformation(vocab_entry),
         complexity_preservation = vocab_entry.complexity,
         semantic_role = vocab_entry.param_type
      }
      
      table.insert(productions, production)
   end
   
   self.hypergraph.grammar_rules[rule_id].productions = productions
   
   -- Add to production system
   self:_categorizeProductions(productions, rule_id)
end

function P9MLCognitiveKernel:_inferTransformation(vocab_entry)
   -- Infer transformation type from vocabulary entry
   local param_type = vocab_entry.param_type
   local shape = vocab_entry.shape
   
   if param_type == 'weight_matrix' then
      return {
         type = 'linear_transformation',
         input_dim = shape[2],
         output_dim = shape[1],
         operation = 'matrix_multiply'
      }
   elseif param_type == 'convolution_kernel' then
      return {
         type = 'convolution_transformation',
         kernel_size = shape,
         operation = 'convolution'
      }
   elseif param_type == 'bias_vector' then
      return {
         type = 'additive_transformation',
         dimension = shape[1],
         operation = 'vector_add'
      }
   else
      return {
         type = 'generic_transformation',
         shape = shape,
         operation = 'element_wise'
      }
   end
end

function P9MLCognitiveKernel:_categorizeProductions(productions, rule_id)
   -- Categorize productions into linguistic categories
   for _, production in ipairs(productions) do
      local category = self:_determineProductionCategory(production)
      
      if not self.productions[category] then
         self.productions[category] = {}
      end
      
      table.insert(self.productions[category], {
         production = production,
         rule_id = rule_id,
         activation_strength = 1.0
      })
   end
end

function P9MLCognitiveKernel:_determineProductionCategory(production)
   -- Determine linguistic category of production
   local transform_type = production.output_transformation.type
   
   if transform_type == 'linear_transformation' then
      return 'syntactic'  -- Structural transformations
   elseif transform_type == 'convolution_transformation' then
      return 'semantic'   -- Feature extraction
   elseif transform_type == 'additive_transformation' then
      return 'lexical'    -- Bias adjustments
   else
      return 'pragmatic'  -- Context-dependent transformations
   end
end

function P9MLCognitiveKernel:addMetaGrammar(namespace_info)
   -- Add namespace as meta-grammar
   local meta_id = 'meta_' .. namespace_info.namespace_id
   
   self.hypergraph.meta_grammar[meta_id] = {
      namespace_id = namespace_info.namespace_id,
      orchestration_rules = namespace_info.orchestration_rules,
      meta_rules = namespace_info.meta_rules,
      hypergraph_stats = namespace_info.hypergraph_stats,
      meta_cognitive_level = self:_calculateMetaCognitiveLevel(namespace_info)
   }
   
   return meta_id
end

function P9MLCognitiveKernel:_calculateMetaCognitiveLevel(namespace_info)
   -- Calculate meta-cognitive level of namespace
   local rule_complexity = namespace_info.orchestration_rules + namespace_info.meta_rules * 2
   local connectivity = namespace_info.hypergraph_stats.edges / math.max(namespace_info.hypergraph_stats.nodes, 1)
   
   return rule_complexity * connectivity
end

function P9MLCognitiveKernel:resolveFrameProblem(context, query_tensor)
   -- Resolve frame problem using nested membrane embeddings
   local context_embedding = self:_generateContextEmbedding(context)
   local query_embedding = self:_generateQueryEmbedding(query_tensor)
   
   -- Find relevant frame embeddings
   local relevant_frames = self:_findRelevantFrames(context_embedding, query_embedding)
   
   -- Generate nested context resolution
   local resolution = self:_generateNestedResolution(relevant_frames, context, query_tensor)
   
   return resolution
end

function P9MLCognitiveKernel:_generateContextEmbedding(context)
   -- Generate embedding for current context
   local embedding = torch.zeros(256)  -- Fixed size embedding
   
   -- Simple hash-based embedding for now
   for key, value in pairs(context) do
      local hash = self:_hashString(tostring(key) .. tostring(value))
      local index = (hash % 256) + 1
      embedding[index] = embedding[index] + 1
   end
   
   return embedding:div(embedding:norm() + 1e-8)  -- Normalize
end

function P9MLCognitiveKernel:_generateQueryEmbedding(query_tensor)
   -- Generate embedding for query tensor
   if torch.isTensor(query_tensor) then
      local flattened = query_tensor:view(-1)
      local embedding = torch.zeros(256)
      
      -- Sample key elements for embedding
      local sample_size = math.min(256, flattened:nElement())
      local indices = torch.randperm(flattened:nElement()):narrow(1, 1, sample_size)
      
      for i = 1, sample_size do
         embedding[i] = flattened[indices[i]]
      end
      
      return embedding:div(embedding:norm() + 1e-8)
   else
      return torch.zeros(256)
   end
end

function P9MLCognitiveKernel:_findRelevantFrames(context_emb, query_emb)
   -- Find relevant frame embeddings
   local relevant_frames = {}
   local threshold = 0.5
   
   for frame_id, frame_data in pairs(self.frame_embeddings) do
      local context_sim = torch.dot(context_emb, frame_data.context_embedding)
      local query_sim = torch.dot(query_emb, frame_data.query_embedding)
      
      local total_sim = (context_sim + query_sim) / 2.0
      
      if total_sim > threshold then
         table.insert(relevant_frames, {
            frame_id = frame_id,
            similarity = total_sim,
            frame_data = frame_data
         })
      end
   end
   
   -- Sort by similarity
   table.sort(relevant_frames, function(a, b) return a.similarity > b.similarity end)
   
   return relevant_frames
end

function P9MLCognitiveKernel:_generateNestedResolution(relevant_frames, context, query_tensor)
   -- Generate nested context resolution
   local resolution = {
      primary_context = context,
      nested_contexts = {},
      resolution_path = {},
      cognitive_coherence = 0.0
   }
   
   for i, frame in ipairs(relevant_frames) do
      if i <= 5 then  -- Limit to top 5 frames
         local nested_context = self:_generateNestedContext(frame.frame_data, context)
         table.insert(resolution.nested_contexts, nested_context)
         table.insert(resolution.resolution_path, frame.frame_id)
      end
   end
   
   resolution.cognitive_coherence = self:_calculateCognitiveCoherence(resolution)
   
   return resolution
end

function P9MLCognitiveKernel:_generateNestedContext(frame_data, parent_context)
   -- Generate nested context from frame data
   local nested = {}
   
   for key, value in pairs(parent_context) do
      nested[key] = value
   end
   
   -- Add frame-specific context
   nested.frame_id = frame_data.frame_id
   nested.frame_level = (parent_context.frame_level or 0) + 1
   nested.inheritance_path = frame_data.inheritance_path or {}
   
   return nested
end

function P9MLCognitiveKernel:_calculateCognitiveCoherence(resolution)
   -- Calculate cognitive coherence of resolution
   local coherence = 0.0
   local context_count = #resolution.nested_contexts
   
   if context_count > 0 then
      coherence = 1.0 / (1.0 + context_count * 0.1)  -- Decreasing coherence with complexity
   end
   
   return coherence
end

function P9MLCognitiveKernel:_hashString(str)
   -- Hash function for string inputs
   local hash = 5381
   for i = 1, #str do
      hash = ((hash * 33) + string.byte(str, i)) % 1000000
   end
   return hash
end

function P9MLCognitiveKernel:generateGestaltField()
   -- Generate unified gestalt tensor field from all components
   local field_components = {}
   
   -- Collect lexical components
   for lexeme_id, lexeme in pairs(self.hypergraph.lexemes) do
      table.insert(field_components, {
         type = 'lexical',
         weight = lexeme.semantic_weight,
         prime_factors = lexeme.prime_factors,
         shape = lexeme.shape
      })
   end
   
   -- Collect grammatical components
   for rule_id, rule in pairs(self.hypergraph.grammar_rules) do
      table.insert(field_components, {
         type = 'grammatical',
         weight = rule.cognitive_strength,
         productions = rule.productions
      })
   end
   
   -- Generate gestalt tensor
   local gestalt_size = math.max(64, #field_components * 4)
   local gestalt_tensor = torch.zeros(gestalt_size, gestalt_size)
   
   for i, component in ipairs(field_components) do
      local row = ((i - 1) % gestalt_size) + 1
      local col = ((math.floor((i - 1) / gestalt_size)) % gestalt_size) + 1
      gestalt_tensor[row][col] = component.weight
   end
   
   self.cognitive_field.gestalt_state = gestalt_tensor
   self.cognitive_field.coherence_measure = self:_calculateFieldCoherence(gestalt_tensor)
   
   return gestalt_tensor
end

function P9MLCognitiveKernel:_calculateFieldCoherence(gestalt_tensor)
   -- Calculate coherence measure of gestalt field
   local eigenvalues = torch.symeig(gestalt_tensor + gestalt_tensor:t())  -- Symmetrize first
   local coherence = eigenvalues:max() / (eigenvalues:sum() + 1e-8)
   return coherence
end

function P9MLCognitiveKernel:getCognitiveState()
   -- Return comprehensive cognitive kernel state
   return {
      lexemes_count = self:_countTableEntries(self.hypergraph.lexemes),
      grammar_rules_count = self:_countTableEntries(self.hypergraph.grammar_rules),
      meta_grammar_count = self:_countTableEntries(self.hypergraph.meta_grammar),
      prime_factors_catalog = self.cognitive_field.prime_factors,
      gestalt_coherence = self.cognitive_field.coherence_measure,
      production_categories = {
         lexical = #self.productions.lexical,
         syntactic = #self.productions.syntactic,
         semantic = #self.productions.semantic,
         pragmatic = #self.productions.pragmatic
      }
   }
end

function P9MLCognitiveKernel:_countTableEntries(table)
   local count = 0
   for _ in pairs(table) do
      count = count + 1
   end
   return count
end

function P9MLCognitiveKernel:__tostring__()
   local state = self:getCognitiveState()
   return string.format('nn.P9MLCognitiveKernel(lexemes=%d, rules=%d, coherence=%.3f)', 
                       state.lexemes_count, state.grammar_rules_count, state.gestalt_coherence)
end