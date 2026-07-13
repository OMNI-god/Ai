using Ai.Models;
using Ai.Models.Regression;
using Ai.NeuralNetworks.Layers;
using Ai.NeuralNetworks.Network;
using Vec = Ai.Math.Vector.Vector;
using Mat = Ai.Math.Matrix.Matrix;
using Ai.NeuralNetworks.Activations;
using Ai.NeuralNetworks.Losses;

namespace Ai.Playground;

public class Program
{
    public static void Main(string[] args)
    {
        Vec a = new Vec(1, 2, 3);
        Vec b = new Vec(4, 5, 6);

        var c = a + b;
        Console.WriteLine(c.ToString());

        var avg = Math.Vector.Vector.Average(a, b);

        Console.WriteLine(avg.ToString());

        // //linear regression

        // var lr = new LinearRegression
        // {
        //     Weights = new Vec(0),
        //     Bias = 0
        // };

        // var samplesLr = new List<TrainingSample>
        // {
        //     new(new Vec(1), 3),
        //     new(new Vec(2), 5),
        //     new(new Vec(3), 7),
        //     new(new Vec(4), 9),
        //     new(new Vec(5), 11),
        // };

        // Console.WriteLine($"Weight : {lr.Weights}");
        // Console.WriteLine($"Bias   : {lr.Bias}");
        // Console.WriteLine($"Cost   : {lr.Cost(samplesLr)}");

        // lr.Train(samplesLr, 10000, 0.01);

        // Console.WriteLine($"Weight : {lr.Weights}");
        // Console.WriteLine($"Bias   : {lr.Bias}");
        // Console.WriteLine($"Cost   : {lr.Cost(samplesLr)}");

        // //Logistis
        // var log = new LogisticRegression
        // {
        //     Weights = new Vec(0, 0),
        //     Bias = 0
        // };
        // var samplesLog = new List<TrainingSample>
        // {
        //     new(new Vec(0,0),0),
        //     new(new Vec(0,1),1),
        //     new(new Vec(1,0),1),
        //     new(new Vec(1,1),1)
        // };

        // Console.WriteLine("Before Training");


        // Console.WriteLine(log.Prediction(new Vec(0, 1)));
        // Console.WriteLine(log.Prediction(new Vec(1, 0)));
        // Console.WriteLine(log.Prediction(new Vec(1, 1)));

        // Console.WriteLine(log.Cost(samplesLog));
        // log.Train(samplesLog, 10000, 0.1);
        // Console.WriteLine("After Training");

        // Console.WriteLine(log.Prediction(new Vec(0, 0)));
        // Console.WriteLine(log.Prediction(new Vec(0, 1)));
        // Console.WriteLine(log.Prediction(new Vec(1, 0)));
        // Console.WriteLine(log.Prediction(new Vec(1, 1)));

        // Console.WriteLine(log.Cost(samplesLog));

        var samples = new List<TrainingSample>
        {
            new(new Vec(1), new Vec(3)),
            new(new Vec(2), new Vec(5)),
            new(new Vec(3), new Vec(7)),
            new(new Vec(4), new Vec(9)),
        };
        var network = new Sequential(
            new DenseLayer(
                new Mat(new double[,]
                {
            { 0.5 }
                }),
                new Vec(0),
                new ReLU()
            )
        );
        var loss = new MeanSquaredError();
        foreach (var sample in samples)
        {
            Console.WriteLine(
                $"{sample.Input} -> {network.Forward(sample.Input)}");
        }

        network.Train(samples, loss, epochs: 5000, learningRate: 0.01);

        Console.WriteLine("After Training");

        foreach (var sample in samples)
        {
            Console.WriteLine(
                $"{sample.Input} -> {network.Forward(sample.Input)}");
        }
    }
}