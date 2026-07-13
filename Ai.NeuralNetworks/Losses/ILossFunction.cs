namespace Ai.NeuralNetworks.Losses;

using Vec = Ai.Math.Vector.Vector;

public interface ILossFunction
{
    double Calculate(Vec prediction, Vec expected);

    Vec Derivative(Vec prediction, Vec expected);
}