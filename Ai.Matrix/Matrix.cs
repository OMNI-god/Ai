using Vec = Ai.Vector.Vector;

namespace Ai.Matrix;

public class Matrix
{
    public readonly double[,] Values;
    public int Rows => Values.GetLength(0);
    public int Columns => Values.GetLength(1);
    public double this[int row, int col] => Values[row, col];

    public Matrix(double[,] Values)
    {
        this.Values = Values;
    }

    public static Matrix operator +(Matrix a, Matrix b)
    {
        IsValid(a, b);

        double[,] c = new double[a.Rows, a.Columns];
        for (int row = 0; row < a.Rows; row++)
        {
            for (int col = 0; col < a.Columns; col++)
            {
                c[row, col] = a[row, col] + b[row, col];
            }
        }
        return new Matrix(c);
    }
    public static Matrix operator -(Matrix a, Matrix b)
    {
        IsValid(a, b);

        double[,] c = new double[a.Rows, a.Columns];
        for (int row = 0; row < a.Rows; row++)
        {
            for (int col = 0; col < a.Columns; col++)
            {
                c[row, col] = a[row, col] - b[row, col];
            }
        }
        return new Matrix(c);
    }
    public static Matrix operator *(Matrix m, double scalar)
    {
        double[,] c = new double[m.Rows, m.Columns];
        for (int row = 0; row < m.Rows; row++)
        {
            for (int col = 0; col < m.Columns; col++)
            {
                c[row, col] = m[row, col] * scalar;
            }
        }
        return new Matrix(c);
    }
    public static Matrix operator /(Matrix m, double scalar)
    {
        if (scalar == 0)
            throw new DivideByZeroException();
        double[,] c = new double[m.Rows, m.Columns];
        for (int row = 0; row < m.Rows; row++)
        {
            for (int col = 0; col < m.Columns; col++)
            {
                c[row, col] = m[row, col] / scalar;
            }
        }
        return new Matrix(c);
    }
    public Matrix Transpose()
    {
        double[,] c = new double[this.Columns, this.Rows];

        for (int row = 0; row < this.Rows; row++)
        {
            for (int col = 0; col < this.Columns; col++)
            {
                c[col, row] = this[row, col];
            }
        }
        return new Matrix(c);
    }

    public Vec Multiply(Vec v)
    {
        if (Columns != v.Length)
            throw new ArgumentException("Matrix columns must equal vector length.");

        double[] result = new double[Rows];

        for (int row = 0; row < Rows; row++)
        {
            for (int col = 0; col < Columns; col++)
            {
                result[row] += this[row, col] * v[col];
            }
        }

        return new Vec(result);
    }

    public Matrix Multiply(Matrix m)
    {
        if (this.Columns != m.Rows)
            throw new ArgumentException("Matrix dimensions are incompatible.");

        double[,] res = new double[this.Rows, m.Columns];

        for (int row = 0; row < this.Rows; row++)
        {
            for (int col = 0; col < m.Columns; col++)
            {
                for (int k = 0; k < this.Columns; k++)
                {
                    res[row, col] += this[row, k] * m[k, col];
                }
            }
        }

        return new Matrix(res);
    }
    public static Matrix Identity(int size)
    {
        double[,] res = new double[size, size];
        for (int row = 0; row < size; row++)
        {
            for (int col = 0; col < size; col++)
            {
                if (row == col) res[row, col] = 1;
            }
        }
        return new Matrix(res);
    }
    private static void IsValid(Matrix a, Matrix b)
    {
        if (a.Rows != b.Rows || a.Columns != b.Columns)
        {
            throw new ArgumentException("Both matrix should be of same dimenssion");
        }
    }
    public override string ToString()
    {
        string res = "";
        for (int row = 0; row < this.Rows; row++)
        {
            res += "[ ";
            for (int col = 0; col < this.Columns; col++)
            {
                res += $"{this[row, col]} ";
                if (col != this.Columns - 1)
                {
                    res += ",";
                }
            }
            res += "]";
            res += "\n";
        }
        return res;
    }
    public Vec GetRow(int row)
    {
        if (row < 0 || row >= Rows)
            throw new ArgumentOutOfRangeException(nameof(row));
        double[] res = new double[this.Columns];
        for (int col = 0; col < this.Columns; col++)
        {
            res[col] = this[row, col];
        }
        return new Vec(res);
    }

    public Vec GetColumn(int col)
    {
        if (col < 0 || col >= this.Columns)
            throw new ArgumentOutOfRangeException(nameof(col));
        double[] res = new double[this.Rows];
        for (int row = 0; row < this.Rows; row++)
        {
            res[row] = this[row, col];
        }
        return new Vec(res);
    }
}
