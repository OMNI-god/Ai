using Ai.NeuralNetworks.Activations;
using Mat = Ai.Matrix.Matrix;
using Vec = Ai.Vector.Vector;

namespace Ai.NeuralNetworks.Layers;

public class DenseLayer : ILayer
{
    public Mat Weights { get; private set; }

    public Vec Bias { get; private set; }
    public IActivationFunction Activation { get; }

    public DenseLayer(Mat Weights, Vec Bias, IActivationFunction Activation)
    {
        this.Weights = Weights;
        this.Bias = Bias;
        this.Activation = Activation;
    }
    public Vec Forward(Vec input)
    {
        return (this.Weights.Multiply(input) + this.Bias).Apply(Activation.Activate);
    }
}