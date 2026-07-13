
namespace Ai.NeuralNetworks.Losses;

public class MeanSquaredError : ILossFunction
{
    public double Calculate(Math.Vector.Vector prediction, Math.Vector.Vector expected)
    {
        var error = prediction - expected;
        return error.Dot(error) / prediction.Length;
    }
    public Math.Vector.Vector Derivative(Math.Vector.Vector prediction, Math.Vector.Vector expected)
    {
        return (prediction - expected) * (2.0 / prediction.Length);
    }
}