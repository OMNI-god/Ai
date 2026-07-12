namespace Ai.Vector
{
    public class Vector
    {
        public readonly double[] Values;

        public int Length => Values.Length;

        public double this[int idx]
        {
            get => Values[idx];
            set => Values[idx] = value;
        }

        public Vector(params double[] Values)
        {
            this.Values = Values;
        }
        public static Vector operator +(Vector a, Vector b)
        {
            IsValid(a, b);

            double[] c = new double[a.Values.Length];

            for (int i = 0; i < a.Values.Length; i++)
            {
                c[i] = a.Values[i] + b.Values[i];
            }

            return new Vector(c);
        }

        public static Vector operator -(Vector a, Vector b)
        {
            IsValid(a, b);

            double[] c = new double[a.Values.Length];

            for (int i = 0; i < a.Values.Length; i++)
            {
                c[i] = a.Values[i] - b.Values[i];
            }

            return new Vector(c);
        }

        public double Dot(Vector other)
        {
            IsValid(this, other);

            double sum = 0;

            for (int i = 0; i < other.Values.Length; i++)
            {
                sum += this.Values[i] * other.Values[i];
            }

            return sum;
        }

        public double Magnitude()
        {
            double sum = 0;

            for (int i = 0; i < Values.Length; i++)
            {
                sum += Values[i] * Values[i];
            }

            return Math.Sqrt(sum);
        }

        public Vector Normalize()
        {
            double mag = this.Magnitude();

            if (mag == 0) throw new InvalidOperationException("Cannot normalize a zero vector.");

            double[] res = new double[this.Length];

            for (int i = 0; i < this.Length; i++)
            {
                res[i] = this.Values[i] / mag;
            }

            return new Vector(res);
        }

        public double CosineSimilarity(Vector other)
        {
            return Dot(other) / (Magnitude() * other.Magnitude());
        }

        public Vector Cross(Vector other)
        {
            IsValid(this, other);

            if (Values.Length != 3 || other.Values.Length != 3)
            {
                throw new ArgumentException("For cross product both should have length of 3");
            }

            return new Vector(
                this[1] * other[2] - this[2] * other[1],
                this[2] * other[0] - this[0] * other[2],
                this[0] * other[1] - this[1] * other[0]
                );
        }

        private static void IsValid(Vector a, Vector b)
        {
            if (a.Values.Length != b.Values.Length)
            {
                throw new ArgumentException("Length of the vectors should be same.");
            }
        }

        public override string ToString()
        {
            return string.Join(" ,", Values);
        }

        public static Vector Average(params Vector[] vectors)
        {
            double[] res = new double[vectors[0].Length];

            for (int i = 0; i < vectors.Length; i++)
            {
                Vector temp = new Vector(res);
                temp += vectors[i];
                res = temp.Values;
            }

            return new Vector(res) / vectors.Length;
        }

        public static Vector operator /(Vector a, double scalar)
        {
            double[] res = new double[a.Length];

            for (int i = 0; i < a.Length; i++)
            {
                res[i] = a[i] / scalar;
            }

            return new Vector(res);
        }
        public static Vector operator *(Vector a, double scalar)
        {
            double[] res = new double[a.Length];

            for (int i = 0; i < a.Length; i++)
            {
                res[i] = a[i] * scalar;
            }

            return new Vector(res);
        }
        public Vector Apply(Func<double, double> function)
        {
            double[] res = new double[this.Length];
            for (int i = 0; i < this.Length; i++)
            {
                res[i] = function(this[i]);
            }
            return new Vector(res);
        }
    }
}
