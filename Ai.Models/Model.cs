using Vec = Ai.Vector.Vector;
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
}