using Vec = Ai.Math.Vector.Vector;
namespace Ai.Models;

public abstract class Model
{
    public Vec Weights { get; set; }
    public double Bias { get; set; }

    protected double CalculateLinearOutput(Vec ip)
    {
        return ip.Dot(Weights) + Bias;
    }
    public abstract double Prediction(Vec ip);
    public abstract Vec CalculateWeightGradient(IEnumerable<TrainingSample> samples);
    public abstract double CalculateBiasGradient(IEnumerable<TrainingSample> samples);

    public void Train(IEnumerable<TrainingSample> samples, int epochs, double learningRate)
    {
        for (int i = 0; i < epochs; i++)
        {
            Weights = Weights - CalculateWeightGradient(samples) * learningRate;
            Bias = Bias - learningRate * CalculateBiasGradient(samples);
        }
    }
}