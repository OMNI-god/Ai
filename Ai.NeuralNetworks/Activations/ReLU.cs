namespace Ai.NeuralNetworks.Activations;

public class ReLU : IActivationFunction
{
    public double Activate(double x)
    {
        if (x <= 0) return 0;
        return x;
    }

    public double Derivative(double x)
    {
        if (x <= 0) return 0;
        return 1;
    }
}