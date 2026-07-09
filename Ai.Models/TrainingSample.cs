using Vec = Ai.Vector.Vector;

namespace Ai.Models;

public class TrainingSample
{
    public Vec Input { get; }

    public double ExpectedOutput { get; }

    public TrainingSample(Vec input, double expectedOutput)
    {
        Input = input;
        ExpectedOutput = expectedOutput;
    }

}