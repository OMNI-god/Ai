using Vec = Ai.Vector.Vector;

namespace Ai.Models.Regression;

public class LogisticRegression : Model
{
    public override double Prediction(Vec ip)
    {
        double prediction = CalculateLinearOutput(ip);
        return Sigmoid(prediction);
    }
    public double Sigmoid(double input)
    {
        return 1 / (1 + Math.Exp(-input));
    }

    //Binary Cross Entropy
    public double Cost(IEnumerable<TrainingSample> samples)
    {
        double sum = 0, count = 0;
        foreach (var sample in samples)
        {
            count++;
            var prediction = Prediction(sample.Input);
            prediction = Clamp(prediction);
            sum += -(sample.ExpectedOutput * Math.Log(prediction) + (1 - sample.ExpectedOutput) * Math.Log(1 - prediction));
        }
        if (count == 0) throw new InvalidOperationException("Training data cannot be empty.");
        return sum / count;
    }
    private double Clamp(double prediction) => Math.Clamp(prediction, 1e-15, 1 - 1e-15);

    public override Vec CalculateWeightGradient(IEnumerable<TrainingSample> samples)
    {
        Vec gradient = new Vec(new double[Weights.Length]);
        double count = 0;
        foreach (var sample in samples)
        {
            count++;
            var prediction = Prediction(sample.Input);
            var error = prediction - sample.ExpectedOutput;
            for (int i = 0; i < Weights.Length; i++)
            {
                gradient[i] += error * sample.Input[i];
            }
        }
        return gradient / count;
    }

    public override double CalculateBiasGradient(IEnumerable<TrainingSample> samples)
    {
        double gradient = 0;
        double count = 0;
        foreach (var sample in samples)
        {
            count++;
            var prediction = Prediction(sample.Input);
            var error = prediction - sample.ExpectedOutput;
            gradient += error;
        }
        return gradient / count;
    }
}