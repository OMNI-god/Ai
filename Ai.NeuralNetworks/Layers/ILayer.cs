namespace Ai.NeuralNetworks.Layers;

public interface ILayer
{
    Math.Vector.Vector Forward(Math.Vector.Vector input);
    Math.Vector.Vector Backward(Math.Vector.Vector outputGradient, double learningRate);
}