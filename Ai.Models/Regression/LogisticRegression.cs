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

    //Binary Cost Error
    public double Cost(double prediction, double actual)
    {
        prediction = Clamp(prediction);
        return -(actual * Math.Log(prediction) + (1 - actual) * Math.Log(1 - prediction));
    }
    private double Clamp(double prediction) => Math.Clamp(prediction, 1e-15, 1 - 1e-15);
}