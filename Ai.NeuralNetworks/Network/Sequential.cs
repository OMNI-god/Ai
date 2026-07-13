using Ai.Models;
using Ai.NeuralNetworks.Layers;
using Ai.NeuralNetworks.Losses;
using Vec = Ai.Math.Vector.Vector;

namespace Ai.NeuralNetworks.Network;

public class Sequential
{
    private readonly List<ILayer> _layers;

    public IReadOnlyList<ILayer> Layers => _layers;

    public Sequential(params ILayer[] layers)
    {
        _layers = new List<ILayer>(layers);
    }

    public Vec Forward(Vec input)
    {
        var output = input;

        foreach (var layer in Layers)
        {
            output = layer.Forward(output);
        }
        return output;
    }
    public Vec Backward(Vec gradient, double learningRate)
    {
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradient = Layers[i].Backward(gradient, learningRate);
        }
        return gradient;
    }
    public void Train(
    IEnumerable<TrainingSample> samples,
    ILossFunction loss,
    int epochs,
    double learningRate)
    {
        for (int i = 0; i < epochs; i++)
        {
            double epochLoss = 0;

            foreach (var sample in samples)
            {
                var prediction = Forward(sample.Input);

                epochLoss += loss.Calculate(prediction, sample.ExpectedOutput);

                var gradient = loss.Derivative(prediction, sample.ExpectedOutput);

                Backward(gradient, learningRate);
            }

            if (i % 100 == 0)
            {
                Console.WriteLine($"Epoch {i}: Loss = {epochLoss / samples.Count()}");
            }
        }
    }
}