using Ai.NeuralNetworks.Activations;
using Mat = Ai.Math.Matrix.Matrix;
using Vec = Ai.Math.Vector.Vector;

namespace Ai.NeuralNetworks.Layers;

public class DenseLayer : ILayer
{
    private Vec _lastInput;

    private Vec _lastZ;

    private Vec _lastOutput;
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
        this._lastInput = input;
        this._lastZ = this.Weights.Multiply(input) + this.Bias;
        this._lastOutput = _lastZ.Apply(this.Activation.Activate);
        return this._lastOutput;
    }

    public Vec Backward(Vec outputGradient, double learningRate)
    {
        Vec activationDerivative = _lastZ.Apply(Activation.Derivative);

        Vec delta = outputGradient.Hadamard(activationDerivative);

        Mat weightGradient = delta.Outer(_lastInput);

        Vec biasGradient = delta;

        Mat oldWeights = Weights;

        Weights = Weights - weightGradient * learningRate;
        Bias = Bias - biasGradient * learningRate;

        return oldWeights.Transpose().Multiply(delta);
    }
}