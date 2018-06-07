#pragma warning disable 0169
#pragma warning disable 0414

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using GeneticLib.Generations;
using GeneticLib.GeneticManager;
using GeneticLib.Genome;
using GeneticLib.Genome.NeuralGenomes;
using GeneticLib.GenomeFactory;
using GeneticLib.GenomeFactory.GenomeProducer;
using GeneticLib.GenomeFactory.GenomeProducer.Breeding;
using GeneticLib.GenomeFactory.GenomeProducer.Breeding.Crossover;
using GeneticLib.GenomeFactory.GenomeProducer.Selection;
using GeneticLib.GenomeFactory.GenomeProducer.Reinsertion;
using GeneticLib.GenomeFactory.Mutation;
using GeneticLib.GenomeFactory.Mutation.NeuralMutations;
using GeneticLib.Neurology;
using GeneticLib.Randomness;
using GeneticLib.Utils.Graph;
using GeneticLib.Utils.NeuralUtils;
using GeneticLib.Utils.Extensions;
using GeneticLib.Neurology.NeuralModels;
using GeneticLib.Neurology.Neurons;
using GeneticLib.Neurology.NeuronValueModifiers;
using GeneticLib.Generations.InitialGeneration;
using GeneticLib.Genome.NeuralGenomes.NetworkOperationBakers;

namespace NeuralNetsLearningToCount
{
    class Program
    {
		private static readonly string pyNeuralNetGraphDrawerPath =
			"../MachineLearningPyGraphUtils/PyNeuralNetDrawer.py";
		private static readonly string pyFitnessGraphPath =
			"../MachineLearningPyGraphUtils/DrawGraph.py";

		int genomesCount = 50;

        float singleSynapseMutChance = 0.2f;
        float singleSynapseMutValue = 3f;

        float allSynapsesMutChance = 0.1f;
        float allSynapsesMutChanceEach = 1f;
        float allSynapsesMutValue = 2f;

        float crossoverPart = 0.80f;
        float reinsertionPart = 0.2f;

        GeneticManagerClassic geneticManager;
        public static int maxIterations = 30000;

        public static bool targetReached = false;
		private static int generationsWithTargetReached = 5;
		private static int generationsWithTargetReachedCount = 0;

		private static int inputs = 3;
              
        static void Main(string[] args)
        {
			NeuralNetDrawer.pyGraphDrawerPath = pyNeuralNetGraphDrawerPath;
            PyDrawGraph.pyGraphDrawerFilePath = pyFitnessGraphPath;

			GARandomManager.Random = new RandomClassic((int)DateTime.Now.Ticks);
			var neuralNetDrawer = new NeuralNetDrawer(false);
            var fitnessCollector = new GraphDataCollector();

            NeuralGenomeToJSONExtension.distBetweenNodes *= 5;
            NeuralGenomeToJSONExtension.randomPosTries = 10;
			NeuralGenomeToJSONExtension.xPadding = 0.03f;
			NeuralGenomeToJSONExtension.yPadding = 0.03f;

			var program = new Program();

			for (var i = 0; i < maxIterations; i++)
			{
				program.Evaluate();

				if (i % 10 == 0)
				{
					var fintessSum = program.geneticManager
											.GenerationManager
											.CurrentGeneration
											.Genomes.Sum(x => x.Fitness);
					var best = program.geneticManager
									  .GenerationManager
									  .CurrentGeneration
									  .BestGenome as NeuralGenome;

					Console.WriteLine("Count:" + best.Neurons.Count());
					//fitnessCollector.Tick(i, best.Fitness);
					Console.WriteLine(String.Format(
						"{0}) Best:{1:0.00} Sum:{2:0.00}",
						i,
						best.Fitness,
						fintessSum));

					neuralNetDrawer.QueueNeuralNetJson(program.GetBestJson());

					if (best.Fitness > -0.01f)
					{
						generationsWithTargetReachedCount++;
						if (generationsWithTargetReachedCount >= generationsWithTargetReached)
						{
							targetReached = true;
       
							for (var j = 0; j < Math.Pow(2, inputs) - 1; j++)
                            {
								best.FeedNeuralNetwork(GetBits(j).Select(x => (float)x).ToArray());
                                var expectedOutput = GetBits(j + 1);

						        var output = BitsToInt(best.Outputs.Select(x => x.Value).ToArray());
                                Console.WriteLine(String.Format("{0:0.00} | {1}", output, j));
                             }
							break;
						}
					}
					else
					{
						generationsWithTargetReachedCount = 0;
					}
				}

                program.Evolve();
            }
            
			//neuralNetDrawer.QueueNeuralNetJson(program.GetBestJson());
			//fitnessCollector.Draw();
        }

		public Program()
		{         
			var model = new NeuralModelBase();
			model.defaultWeightInitializer = () => GARandomManager.NextFloat(-1f, 1f);
			model.WeightConstraints = new Tuple<float, float>(-20, 20);

			var bias = model.AddBiasNeuron();         
			var layers = new List<Neuron[]>()
			{
				model.AddInputNeurons(inputs).ToArray(),
    
				model.AddNeurons(
					new Neuron(-1, ActivationFunctions.TanH)
    				{
    					//ValueModifiers = new[] { Dropout.DropoutFunc(0.06f) },
    				},
					count: 5
				).ToArray(),

				model.AddNeurons(
                    new Neuron(-1, ActivationFunctions.TanH)
                    {
                        //ValueModifiers = new[] { Dropout.DropoutFunc(0.06f) },
                    },
                    count: 5
                ).ToArray(),

				model.AddOutputNeurons(
                    inputs,
                    ActivationFunctions.Sigmoid
                ).ToArray(),
			};

			model.ConnectLayers(layers);
			model.ConnectBias(bias, layers.Skip(1));

			var synapseTracker = new SynapseInnovNbTracker();

			var initialGenerationGenerator = new NeuralInitialGenerationCreatorBase(
				model,
				new RecursiveNetworkOpBaker());
			
			//var selection = new EliteSelection();
			var selection = new RouletteWheelSelection();
			var crossover = new OnePointCrossover(true);
            var breeding = new BreedingClassic(
                crossoverPart,
                1,
                selection,
                crossover,
                InitMutations()
            );

			var reinsertion = new ReinsertionFromSelection(
				reinsertionPart,
				0,
				new EliteSelection());
            var producers = new IGenomeProducer[] { breeding, reinsertion };
            var genomeForge = new GenomeForge(producers);

            var generationManager = new GenerationManagerKeepLast();
            geneticManager = new GeneticManagerClassic(
                generationManager,
                initialGenerationGenerator,
                genomeForge,
                genomesCount
            );

            geneticManager.Init();
		}
        
		public void Evolve()
        {
            geneticManager.Evolve();
        }

        public void Evaluate()
        {
            var genomes = geneticManager.GenerationManager
                                        .CurrentGeneration
                                        .Genomes;

            foreach (var genome in genomes)
                genome.Fitness = ComputeFitness(genome as NeuralGenome);

            var orderedGenomes = genomes.OrderByDescending(g => g.Fitness)
                                        .ToArray();

            geneticManager.GenerationManager
                          .CurrentGeneration
                          .Genomes = orderedGenomes;
        }

        private float ComputeFitness(NeuralGenome genome)
        {
            var fitness = 0d;
			for (var i = 0; i < Math.Pow(2, inputs) - 1; i++)
			{
				genome.FeedNeuralNetwork(GetBits(i).Select(x => (float)x).ToArray());
				var expectedOutput = GetBits(i + 1);
				fitness -= genome.Outputs.Select(x => x.Value)
				                 .Zip(expectedOutput, (o, e) => Math.Abs(o - e))
								 .Sum();
			}
            
            return (float)fitness;
        }

		private static byte[] GetBits(int value)
		{
			var b = new BitArray(new int[] { value });
			var bits = new bool[b.Count];
			b.CopyTo(bits, 0);
			return bits.Take(inputs).Select(bit => (byte)(bit ? 1 : 0)).ToArray();
		}

		private static float BitsToInt(float[] bits)
		{
			var sum = 0f;
			for (int i = 0; i < bits.Length; i++)
			{
				sum += bits[i] * MathF.Pow(2, i);
			}

			return sum;
		}

        private MutationManager InitMutations()
        {
            var result = new MutationManager();

            result.MutationEntries.Add(new MutationEntry(
                new SingleSynapseWeightMutation(() => singleSynapseMutValue),
                singleSynapseMutChance,
                EMutationType.Independent
            ));

			result.MutationEntries.Add(new MutationEntry(
                new SingleSynapseWeightMutation(() => singleSynapseMutValue * 3),
                singleSynapseMutChance / 20,
                EMutationType.Independent
            ));

            result.MutationEntries.Add(new MutationEntry(
                new AllSynapsesWeightMutation(
                    () => allSynapsesMutValue,
                    allSynapsesMutChanceEach),
                allSynapsesMutChance,
                EMutationType.Independent
            ));

            return result;
        }

		public string GetBestJson()
		{
			var best = geneticManager.GenerationManager
			                         .CurrentGeneration
			                         .BestGenome as NeuralGenome;
			return best.ToJson(
				neuronRadius: 0.02f,
				maxWeight: 10,
				edgeWidth: 1f);
		}
    }
}
