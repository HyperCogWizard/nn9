[![Build Status](https://travis-ci.org/torch/nn.svg?branch=master)](https://travis-ci.org/torch/nn)
<a name="nn.dok"></a>
# Neural Network Package #

This package provides an easy and modular way to build and train simple or complex neural networks using [Torch](https://github.com/torch/torch7/blob/master/README.md).

## P9ML Membrane Computing System Integration

🧠 **NEW**: This package now includes a revolutionary P9ML Membrane Computing System that establishes agentic cognitive grammar foundations for neural networks.

### Key Features:
- **Membrane-embedded neural layers** with cognitive capabilities and evolution rules
- **Distributed namespace management** for global state coordination  
- **Cognitive grammar kernel** with hypergraph representation and prime factor tensor catalogs
- **Quantization Aware Training (QAT)** with data-free adaptive precision
- **Meta-learning loops** for recursive adaptation and self-modification
- **Frame problem resolution** through nested membrane embeddings
- **Gestalt tensor fields** for unified cognitive representation

### Quick Start with P9ML:
```lua
-- Wrap existing layers in P9ML membranes
local membrane = nn.P9MLMembrane(nn.Linear(10, 5), 'cognitive_layer')
membrane:addEvolutionRule(nn.P9MLEvolutionFactory.createGradientEvolution())
membrane:enableQuantization(8, 0.1)

-- Create distributed namespace
local namespace = nn.P9MLNamespace('neural_system')
namespace:registerMembrane(membrane)

-- Build cognitive grammar
local kernel = nn.P9MLCognitiveKernel()
kernel:addLexeme({10, 5}, 'cognitive_layer')
kernel:addGrammarRule(membrane:getMembraneInfo())
```

See [P9ML Integration Documentation](doc/p9ml_integration.md) for comprehensive usage examples.

## Core Neural Network Components

 * Modules are the bricks used to build neural networks. Each are themselves neural networks, but can be combined with other networks using containers to create complex neural networks:
   * [Module](doc/module.md#nn.Module): abstract class inherited by all modules;
   * [Containers](doc/containers.md#nn.Containers): composite and decorator classes like [`Sequential`](doc/containers.md#nn.Sequential), [`Parallel`](doc/containers.md#nn.Parallel), [`Concat`](doc/containers.md#nn.Concat) and [`NaN`](doc/containers.md#nn.NaN);
   * [Transfer functions](doc/transfer.md#nn.transfer.dok): non-linear functions like [`Tanh`](doc/transfer.md#nn.Tanh) and [`Sigmoid`](doc/transfer.md#nn.Sigmoid);
   * [Simple layers](doc/simple.md#nn.simplelayers.dok): like [`Linear`](doc/simple.md#nn.Linear), [`Mean`](doc/simple.md#nn.Mean), [`Max`](doc/simple.md#nn.Max) and [`Reshape`](doc/simple.md#nn.Reshape);
   * [Table layers](doc/table.md#nn.TableLayers): layers for manipulating `table`s like [`SplitTable`](doc/table.md#nn.SplitTable), [`ConcatTable`](doc/table.md#nn.ConcatTable) and [`JoinTable`](doc/table.md#nn.JoinTable);
   * [Convolution layers](doc/convolution.md#nn.convlayers.dok): [`Temporal`](doc/convolution.md#nn.TemporalModules),  [`Spatial`](doc/convolution.md#nn.SpatialModules) and [`Volumetric`](doc/convolution.md#nn.VolumetricModules) convolutions;
 * Criterions compute a gradient according to a given loss function given an input and a target:
   * [Criterions](doc/criterion.md#nn.Criterions): a list of all criterions, including [`Criterion`](doc/criterion.md#nn.Criterion), the abstract class;
   * [`MSECriterion`](doc/criterion.md#nn.MSECriterion): the Mean Squared Error criterion used for regression;
   * [`ClassNLLCriterion`](doc/criterion.md#nn.ClassNLLCriterion): the Negative Log Likelihood criterion used for classification;
 * Additional documentation:
   * [Overview](doc/overview.md#nn.overview.dok) of the package essentials including modules, containers and training;
   * [Training](doc/training.md#nn.traningneuralnet.dok): how to train a neural network using [`StochasticGradient`](doc/training.md#nn.StochasticGradient);
   * [Testing](doc/testing.md): how to test your modules.
   * [Experimental Modules](https://github.com/clementfarabet/lua---nnx/blob/master/README.md): a package containing experimental modules and criteria.
