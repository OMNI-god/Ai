namespace Ai.NeuralNetworks.Layers;

public interface ILayer
{
    Vector.Vector Forward(Vector.Vector input);
}