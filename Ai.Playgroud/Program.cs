using Ai.Models;
using Ai.Models.Regression;
using Vec = Ai.Vector.Vector;

namespace Ai.Playground;

public class Program
{
    public static void Main(string[] args)
    {
        Vec a = new Vec(1, 2, 3);
        Vec b = new Vec(4, 5, 6);

        var c = a + b;
        Console.WriteLine(c.ToString());

        var avg = Vector.Vector.Average(a, b);

        Console.WriteLine(avg.ToString());

        //linear regression

        var lr = new LinearRegression
        {
            Weights = new Vec(0),
            Bias = 0
        };

        var samples = new List<TrainingSample>
        {
            new(new Vec(1), 3),
            new(new Vec(2), 5),
            new(new Vec(3), 7),
            new(new Vec(4), 9),
            new(new Vec(5), 11),
        };

        Console.WriteLine($"Weight : {lr.Weights}");
        Console.WriteLine($"Bias   : {lr.Bias}");
        Console.WriteLine($"Cost   : {lr.Cost(samples)}");

        lr.Train(samples, 10000, 0.01);

        Console.WriteLine($"Weight : {lr.Weights}");
        Console.WriteLine($"Bias   : {lr.Bias}");
        Console.WriteLine($"Cost   : {lr.Cost(samples)}");

        Console.WriteLine(lr.Prediction(new Vec(10)));
    }
}