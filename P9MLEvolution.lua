-- P9ML Evolution Rule System
-- Implements membrane evolution and quantization aware training rules
local P9MLEvolutionRule = torch.class('nn.P9MLEvolutionRule')

function P9MLEvolutionRule:__init(rule_type, parameters)
   self.rule_type = rule_type or 'basic_evolution'
   self.parameters = parameters or {}
   self.activation_count = 0
   self.success_rate = 0.0
   self.adaptation_history = {}
end

function P9MLEvolutionRule:apply(membrane_objects)
   -- Apply evolution rule to membrane objects
   local success_count = 0
   local total_applications = 0
   
   for obj_id, membrane_obj in pairs(membrane_objects) do
      if self:_shouldApplyTo(membrane_obj) then
         local success = self:_applyRuleToObject(membrane_obj)
         total_applications = total_applications + 1
         if success then
            success_count = success_count + 1
         end
      end
   end
   
   -- Update rule statistics
   self.activation_count = self.activation_count + 1
   if total_applications > 0 then
      local current_success_rate = success_count / total_applications
      self.success_rate = (self.success_rate * (self.activation_count - 1) + current_success_rate) / self.activation_count
   end
   
   -- Record adaptation
   self:_recordAdaptation(total_applications, success_count)
end

function P9MLEvolutionRule:_shouldApplyTo(membrane_obj)
   -- Determine if rule should apply to specific membrane object
   if self.rule_type == 'gradient_evolution' then
      return membrane_obj.gradient and membrane_obj.gradient:norm() > 0.01
   elseif self.rule_type == 'weight_decay' then
      return membrane_obj.tensor and membrane_obj.evolution_state ~= 'stable'
   elseif self.rule_type == 'quantum_fluctuation' then
      return membrane_obj.quantum_state and membrane_obj.quantum_state.coherence_factor > 0.5
   elseif self.rule_type == 'adaptive_quantization' then
      return membrane_obj.tensor and membrane_obj.tensor:nElement() > 100
   else
      return true  -- Basic evolution applies to all
   end
end

function P9MLEvolutionRule:_applyRuleToObject(membrane_obj)
   -- Apply specific rule to membrane object
   if self.rule_type == 'gradient_evolution' then
      return self:_applyGradientEvolution(membrane_obj)
   elseif self.rule_type == 'weight_decay' then
      return self:_applyWeightDecay(membrane_obj)
   elseif self.rule_type == 'quantum_fluctuation' then
      return self:_applyQuantumFluctuation(membrane_obj)
   elseif self.rule_type == 'adaptive_quantization' then
      return self:_applyAdaptiveQuantization(membrane_obj)
   elseif self.rule_type == 'membrane_fusion' then
      return self:_applyMembraneFusion(membrane_obj)
   elseif self.rule_type == 'cognitive_adaptation' then
      return self:_applyCognitiveAdaptation(membrane_obj)
   else
      return self:_applyBasicEvolution(membrane_obj)
   end
end

function P9MLEvolutionRule:_applyGradientEvolution(membrane_obj)
   -- Apply gradient-based evolution
   if not membrane_obj.gradient then return false end
   
   local evolution_factor = self.parameters.evolution_rate or 0.01
   local momentum = self.parameters.momentum or 0.9
   
   -- Apply momentum-based evolution
   if not membrane_obj._gradient_momentum then
      membrane_obj._gradient_momentum = torch.zeros(membrane_obj.gradient:size())
   end
   
   membrane_obj._gradient_momentum:mul(momentum):add(membrane_obj.gradient:mul(1 - momentum))
   
   -- Evolve tensor based on gradient momentum
   local evolution_step = membrane_obj._gradient_momentum:clone():mul(evolution_factor)
   membrane_obj.tensor:add(evolution_step)
   
   -- Update quantum state
   if membrane_obj.quantum_state then
      membrane_obj.quantum_state.coherence_factor = membrane_obj.quantum_state.coherence_factor * 0.99
      membrane_obj.quantum_state.superposition:add(evolution_step:mul(0.1))
   end
   
   return true
end

function P9MLEvolutionRule:_applyWeightDecay(membrane_obj)
   -- Apply weight decay evolution
   local decay_rate = self.parameters.decay_rate or 0.0001
   local selective_decay = self.parameters.selective_decay or false
   
   if selective_decay then
      -- Apply selective decay based on gradient magnitude
      if membrane_obj.gradient then
         local grad_mask = torch.gt(torch.abs(membrane_obj.gradient), 0.001)
         local decay_mask = grad_mask:typeAs(membrane_obj.tensor):mul(decay_rate)
         membrane_obj.tensor:cmul(torch.ones(membrane_obj.tensor:size()):add(-decay_mask))
      end
   else
      -- Standard weight decay
      membrane_obj.tensor:mul(1 - decay_rate)
   end
   
   return true
end

function P9MLEvolutionRule:_applyQuantumFluctuation(membrane_obj)
   -- Apply quantum-inspired fluctuations
   if not membrane_obj.quantum_state then return false end
   
   local fluctuation_strength = self.parameters.fluctuation_strength or 0.001
   local coherence_threshold = self.parameters.coherence_threshold or 0.1
   
   if membrane_obj.quantum_state.coherence_factor > coherence_threshold then
      -- Apply quantum fluctuations
      local fluctuation = torch.randn(membrane_obj.tensor:size()):mul(fluctuation_strength)
      membrane_obj.tensor:add(fluctuation)
      
      -- Update superposition
      membrane_obj.quantum_state.superposition:add(fluctuation:mul(0.5))
      
      -- Decoherence
      membrane_obj.quantum_state.coherence_factor = membrane_obj.quantum_state.coherence_factor * 0.95
      
      return true
   end
   
   return false
end

function P9MLEvolutionRule:_applyAdaptiveQuantization(membrane_obj)
   -- Apply adaptive quantization based on tensor statistics
   local target_bits = self.parameters.target_bits or 8
   local adaptation_rate = self.parameters.adaptation_rate or 0.1
   
   -- Calculate tensor statistics
   local tensor_stats = self:_calculateTensorStats(membrane_obj.tensor)
   
   -- Determine optimal quantization parameters
   local optimal_scale = self:_calculateOptimalScale(tensor_stats, target_bits)
   
   -- Apply quantization
   local quantized_tensor = self:_quantizeTensor(membrane_obj.tensor, optimal_scale, target_bits)
   
   -- Gradual adaptation to quantized values
   membrane_obj.tensor:mul(1 - adaptation_rate):add(quantized_tensor:mul(adaptation_rate))
   
   -- Update quantization state
   if not membrane_obj.quantization_state then
      membrane_obj.quantization_state = {}
   end
   membrane_obj.quantization_state.current_scale = optimal_scale
   membrane_obj.quantization_state.current_bits = target_bits
   membrane_obj.quantization_state.adaptation_count = (membrane_obj.quantization_state.adaptation_count or 0) + 1
   
   return true
end

function P9MLEvolutionRule:_calculateTensorStats(tensor)
   -- Calculate comprehensive tensor statistics
   return {
      mean = tensor:mean(),
      std = tensor:std(),
      min = tensor:min(),
      max = tensor:max(),
      sparsity = torch.eq(tensor, 0):sum() / tensor:nElement(),
      kurtosis = self:_calculateKurtosis(tensor)
   }
end

function P9MLEvolutionRule:_calculateKurtosis(tensor)
   -- Calculate kurtosis (fourth moment) of tensor
   local mean = tensor:mean()
   local centered = tensor - mean
   local variance = centered:pow(2):mean()
   local fourth_moment = centered:pow(4):mean()
   
   if variance > 1e-8 then
      return fourth_moment / (variance * variance) - 3.0
   else
      return 0.0
   end
end

function P9MLEvolutionRule:_calculateOptimalScale(stats, target_bits)
   -- Calculate optimal quantization scale
   local dynamic_range = stats.max - stats.min
   local max_quantized_value = math.pow(2, target_bits - 1) - 1
   
   local scale = dynamic_range / (2 * max_quantized_value)
   
   -- Adjust for kurtosis (heavy tails need larger scale)
   local kurtosis_adjustment = 1.0 + math.abs(stats.kurtosis) * 0.1
   scale = scale * kurtosis_adjustment
   
   return math.max(scale, 1e-8)  -- Prevent zero scale
end

function P9MLEvolutionRule:_quantizeTensor(tensor, scale, bits)
   -- Quantize tensor with given scale and bit precision
   local max_val = math.pow(2, bits - 1) - 1
   local min_val = -max_val
   
   local quantized = tensor:clone()
   quantized:div(scale)
   quantized:round()
   quantized:clamp(min_val, max_val)
   quantized:mul(scale)
   
   return quantized
end

function P9MLEvolutionRule:_applyMembraneFusion(membrane_obj)
   -- Apply membrane fusion evolution (requires multiple objects)
   -- This is a placeholder for inter-membrane evolution
   if not membrane_obj._fusion_candidates then
      membrane_obj._fusion_candidates = {}
   end
   
   local fusion_probability = self.parameters.fusion_probability or 0.01
   
   if torch.uniform() < fusion_probability then
      -- Mark for potential fusion
      membrane_obj._fusion_marked = true
      return true
   end
   
   return false
end

function P9MLEvolutionRule:_applyCognitiveAdaptation(membrane_obj)
   -- Apply cognitive adaptation based on usage patterns
   local adaptation_strength = self.parameters.adaptation_strength or 0.01
   local memory_factor = self.parameters.memory_factor or 0.9
   
   -- Initialize cognitive state if not present
   if not membrane_obj.cognitive_state then
      membrane_obj.cognitive_state = {
         usage_history = {},
         adaptation_trace = torch.zeros(membrane_obj.tensor:size()),
         cognitive_weight = 1.0
      }
   end
   
   -- Update usage history
   local current_activity = membrane_obj.gradient and membrane_obj.gradient:norm() or 0.0
   table.insert(membrane_obj.cognitive_state.usage_history, current_activity)
   
   -- Limit history size
   if #membrane_obj.cognitive_state.usage_history > 100 then
      table.remove(membrane_obj.cognitive_state.usage_history, 1)
   end
   
   -- Calculate cognitive adaptation
   local avg_activity = self:_calculateAverageActivity(membrane_obj.cognitive_state.usage_history)
   local adaptation_direction = (current_activity > avg_activity) and 1.0 or -1.0
   
   -- Update adaptation trace
   membrane_obj.cognitive_state.adaptation_trace:mul(memory_factor)
   if membrane_obj.gradient then
      membrane_obj.cognitive_state.adaptation_trace:add(
         membrane_obj.gradient:clone():mul(adaptation_direction * adaptation_strength)
      )
   end
   
   -- Apply cognitive adaptation
   membrane_obj.tensor:add(membrane_obj.cognitive_state.adaptation_trace:mul(0.1))
   
   -- Update cognitive weight
   membrane_obj.cognitive_state.cognitive_weight = 
     membrane_obj.cognitive_state.cognitive_weight * 0.99 + avg_activity * 0.01
   
   return true
end

function P9MLEvolutionRule:_calculateAverageActivity(usage_history)
   -- Calculate average activity from usage history
   if #usage_history == 0 then return 0.0 end
   
   local sum = 0.0
   for _, activity in ipairs(usage_history) do
      sum = sum + activity
   end
   
   return sum / #usage_history
end

function P9MLEvolutionRule:_applyBasicEvolution(membrane_obj)
   -- Apply basic evolution rule
   local evolution_rate = self.parameters.evolution_rate or 0.001
   local noise_factor = self.parameters.noise_factor or 0.01
   
   -- Add small random perturbation
   local perturbation = torch.randn(membrane_obj.tensor:size()):mul(noise_factor)
   membrane_obj.tensor:add(perturbation:mul(evolution_rate))
   
   return true
end

function P9MLEvolutionRule:_recordAdaptation(applications, successes)
   -- Record adaptation statistics
   local adaptation_record = {
      timestamp = os.time(),
      applications = applications,
      successes = successes,
      success_rate = applications > 0 and successes / applications or 0.0
   }
   
   table.insert(self.adaptation_history, adaptation_record)
   
   -- Limit history size
   if #self.adaptation_history > 1000 then
      table.remove(self.adaptation_history, 1)
   end
end

function P9MLEvolutionRule:getEvolutionStats()
   -- Return evolution rule statistics
   return {
      rule_type = self.rule_type,
      activation_count = self.activation_count,
      success_rate = self.success_rate,
      parameters = self.parameters,
      adaptation_history_size = #self.adaptation_history
   }
end

function P9MLEvolutionRule:adaptParameters(feedback)
   -- Adapt rule parameters based on feedback
   if feedback.success_rate then
      if feedback.success_rate < 0.3 then
         -- Poor performance, adjust parameters
         if self.rule_type == 'gradient_evolution' then
            self.parameters.evolution_rate = (self.parameters.evolution_rate or 0.01) * 0.8
         elseif self.rule_type == 'weight_decay' then
            self.parameters.decay_rate = (self.parameters.decay_rate or 0.0001) * 0.9
         elseif self.rule_type == 'quantum_fluctuation' then
            self.parameters.fluctuation_strength = (self.parameters.fluctuation_strength or 0.001) * 0.7
         end
      elseif feedback.success_rate > 0.8 then
         -- Good performance, increase aggressiveness
         if self.rule_type == 'gradient_evolution' then
            self.parameters.evolution_rate = (self.parameters.evolution_rate or 0.01) * 1.1
         elseif self.rule_type == 'adaptive_quantization' then
            self.parameters.adaptation_rate = math.min(0.5, (self.parameters.adaptation_rate or 0.1) * 1.2)
         end
      end
   end
end

function P9MLEvolutionRule:__tostring__()
   return string.format('nn.P9MLEvolutionRule(type=%s, activations=%d, success_rate=%.3f)', 
                       self.rule_type, self.activation_count, self.success_rate)
end

-- Factory function for creating common evolution rules
local P9MLEvolutionFactory = {}

function P9MLEvolutionFactory.createGradientEvolution(evolution_rate, momentum)
   return nn.P9MLEvolutionRule('gradient_evolution', {
      evolution_rate = evolution_rate or 0.01,
      momentum = momentum or 0.9
   })
end

function P9MLEvolutionFactory.createWeightDecay(decay_rate, selective)
   return nn.P9MLEvolutionRule('weight_decay', {
      decay_rate = decay_rate or 0.0001,
      selective_decay = selective or false
   })
end

function P9MLEvolutionFactory.createQuantumFluctuation(strength, threshold)
   return nn.P9MLEvolutionRule('quantum_fluctuation', {
      fluctuation_strength = strength or 0.001,
      coherence_threshold = threshold or 0.1
   })
end

function P9MLEvolutionFactory.createAdaptiveQuantization(bits, adaptation_rate)
   return nn.P9MLEvolutionRule('adaptive_quantization', {
      target_bits = bits or 8,
      adaptation_rate = adaptation_rate or 0.1
   })
end

function P9MLEvolutionFactory.createCognitiveAdaptation(strength, memory_factor)
   return nn.P9MLEvolutionRule('cognitive_adaptation', {
      adaptation_strength = strength or 0.01,
      memory_factor = memory_factor or 0.9
   })
end

-- Export factory
nn.P9MLEvolutionFactory = P9MLEvolutionFactory