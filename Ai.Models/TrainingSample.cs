using Vec = Ai.Math.Vector.Vector;

namespace Ai.Models;

public class TrainingSample
{
    public Vec Input { get; }

    public Vec ExpectedOutput { get; }

    public TrainingSample(Vec input, Vec expectedOutput)
    {
        Input = input;
        ExpectedOutput = expectedOutput;
    }

}