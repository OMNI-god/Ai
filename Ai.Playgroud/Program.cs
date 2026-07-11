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
        var log = new LogisticRegression
        {
            Weights = new Vec(0, 0),
            Bias = 0
        };

        var samplesLr = new List<TrainingSample>
        {
            new(new Vec(1), 3),
            new(new Vec(2), 5),
            new(new Vec(3), 7),
            new(new Vec(4), 9),
            new(new Vec(5), 11),
        };

        Console.WriteLine($"Weight : {lr.Weights}");
        Console.WriteLine($"Bias   : {lr.Bias}");
        Console.WriteLine($"Cost   : {lr.Cost(samplesLr)}");

        lr.Train(samplesLr, 10000, 0.01);

        Console.WriteLine($"Weight : {lr.Weights}");
        Console.WriteLine($"Bias   : {lr.Bias}");
        Console.WriteLine($"Cost   : {lr.Cost(samplesLr)}");

        var samplesLog = new List<TrainingSample>
        {
            new(new Vec(0,0),0),
            new(new Vec(0,1),1),
            new(new Vec(1,0),1),
            new(new Vec(1,1),1)
        };

        Console.WriteLine("Before Training");


        Console.WriteLine(log.Prediction(new Vec(0, 1)));
        Console.WriteLine(log.Prediction(new Vec(1, 0)));
        Console.WriteLine(log.Prediction(new Vec(1, 1)));

        Console.WriteLine(log.Cost(samplesLog));
        log.Train(samplesLog, 10000, 0.1);
        Console.WriteLine("After Training");

        Console.WriteLine(log.Prediction(new Vec(0, 0)));
        Console.WriteLine(log.Prediction(new Vec(0, 1)));
        Console.WriteLine(log.Prediction(new Vec(1, 0)));
        Console.WriteLine(log.Prediction(new Vec(1, 1)));

        Console.WriteLine(log.Cost(samplesLog));

    }
}