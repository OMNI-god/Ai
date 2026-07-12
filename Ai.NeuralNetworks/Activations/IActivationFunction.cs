namespace Ai.NeuralNetworks.Activations;

public interface IActivationFunction
{
    double Activate(double x);
    double Derivative(double x);
}