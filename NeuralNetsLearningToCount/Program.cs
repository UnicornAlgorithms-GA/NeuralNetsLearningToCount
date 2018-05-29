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
using GeneticLib.GenomeFactory.GenomeProducer.Breeding.Selection;
using GeneticLib.GenomeFactory.GenomeProducer.Reinsertion;
using GeneticLib.GenomeFactory.Mutation;
using GeneticLib.GenomeFactory.Mutation.NeuralMutations;
using GeneticLib.Neurology;
using GeneticLib.Randomness;
using GeneticLib.Utils.Graph;
using GeneticLib.Utils.NeuralUtils;
using GeneticLib.Utils.Extensions;

namespace NeuralNetsLearningToCount
{
    class Program
    {
		int genomesCount = 50;

        float singleSynapseMutChance = 0.2f;
        float singleSynapseMutValue = 3f;

        float allSynapsesMutChance = 0.1f;
        float allSynapsesMutChanceEach = 1f;
        float allSynapsesMutValue = 1f;

        float crossoverPart = 0.80f;
        float reinsertionPart = 0.2f;

        GeneticManagerClassic geneticManager;
        public static int maxIterations = 30000;
        public static bool targetReached = false;

		int inputs = 3;
              
        static void Main(string[] args)
        {
			GARandomManager.Random = new RandomClassic((int)DateTime.Now.Ticks);
			var neuralNetDrawer = new NeuralNetDrawer(false);
            var fitnessCollector = new GraphDataCollector();

            NeuralGenomeToJSONExtension.distBetweenNodes *= 5;
            NeuralGenomeToJSONExtension.randomPosTries = 10;

			var program = new Program();

            for (var i = 0; i < maxIterations; i++)
            {
                if (targetReached)
                    break;

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
					fitnessCollector.Tick(i, best.Fitness);
					Console.WriteLine(String.Format(
						"{0}) Best:{1:0.00} Sum:{2:0.00}",
						i,
						best.Fitness,
						fintessSum));

					if (i % 100 == 0)
						neuralNetDrawer.QueueNeuralNetJson(program.GetBestJson());
				}

                program.Evolve();
            }

            fitnessCollector.Draw();
			neuralNetDrawer.QueueNeuralNetJson(program.GetBestJson());
        }

		public Program()
		{
			var synapseTracker = new SynapseInnovNbTracker();

			var initialGenerationGenerator = new NIGCLearningToCount(
                synapseTracker,
				inputs,
				inputs,
                new[] { 6 },
                () => (float)GARandomManager.Random.NextDouble(-1, 1),
                true
			);

            var selection = new EliteSelection();
            var crossover = new OnePointCrossover(true);
            var breeding = new BreedingClassic(
                crossoverPart,
                1,
                selection,
                crossover,
                InitMutations()
            );

            var reinsertion = new EliteReinsertion(reinsertionPart, 0);
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

			if (Math.Abs(fitness) < 0.01)
				targetReached = true;

            return (float)fitness;
        }

		private byte[] GetBits(int value)
		{
			var b = new BitArray(new int[] { value });
			var bits = new bool[b.Count];
			b.CopyTo(bits, 0);
			return bits.Take(inputs).Select(bit => (byte)(bit ? 1 : 0)).ToArray();
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
                singleSynapseMutChance / 40,
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
