using Vec = Ai.Vector.Vector;

namespace Ai.Models.Regression;

public class LinearRegression : Model
{
    public override double Prediction(Vec ip)
    {
        return CalculateLinearOutput(ip);
    }

    // Mean Square Error
    public double Cost(IEnumerable<TrainingSample> trainingSamples)
    {
        double sum = 0;
        int count = 0;

        foreach (var sample in trainingSamples)
        {
            var prediction = Prediction(sample.Input);
            double error = prediction - sample.ExpectedOutput;
            sum += error * error;
            count++;
        }
        if (count == 0) throw new InvalidOperationException("Training data cannot be empty.");
        return sum / count;
    }
    public Vec CalculateGradient(IEnumerable<TrainingSample> samples)
    {
        Vec orginalWeights = new Vec(Weights.Values);
        double orginalCost = Cost(samples);

        Vec gradient = new Vec(new double[Weights.Length]);

        double epsilon = 0.00001;

        for (int i = 0; i < Weights.Length; i++)
        {
            double[] newWeights = new double[Weights.Length];
            newWeights = Weights.Values;
            newWeights[i] += epsilon;

            Weights = new Vec(newWeights);
            double newCost = Cost(samples);

            gradient.Values[i] = (newCost - orginalCost) / epsilon;
        }

        Weights = orginalWeights;

        return gradient;
    }

    public Vec CalculateWeightGradient(IEnumerable<TrainingSample> samples)
    {
        Vec gradient = new Vec(new double[Weights.Length]);

        foreach (var sample in samples)
        {
            var prediction = Prediction(sample.Input);
            var error = prediction - sample.ExpectedOutput;

            for (int i = 0; i < Weights.Length; i++)
            {
                gradient[i] += error * sample.Input[i];
            }
        }
        return gradient * (2.0 / samples.Count());
    }

    public double CalculateBiasGradient(IEnumerable<TrainingSample> samples)
    {
        double gradient = 0;
        foreach (var sample in samples)
        {
            var prediction = Prediction(sample.Input);
            var error = prediction - sample.ExpectedOutput;

            gradient += error;
        }
        return gradient * (2.0 / samples.Count());
    }

    public void Train(IEnumerable<TrainingSample> samples, int epochs, double learningRate)
    {
        for (int i = 0; i < epochs; i++)
        {
            Weights = Weights - CalculateWeightGradient(samples) * learningRate;
            Bias = Bias - learningRate * CalculateBiasGradient(samples);
        }
    }
}