Matrix matrix = new Matrix(2,3)

public static Tuple<Matrix, Matrix> GramSchmidt(this Matrix a)
{
    var m = a.M_Rows;
    var n = a.N_Cols;

    Vector[] Q = new Vector[a.N_Cols];
    Matrix R = new Matrix(a.M_Rows, a.N_Cols);

    // This loop calculates r11 and q1
    for (var i = 0; i < n; i++) {
        R[i, i] = a.Column(i).TwoNorm();
        Q[i] = a.Column(i).Normalize();

        for (var j = i + 1; j < n; j++) {
            R[j, i] = Q[i].Dot(a.Column(j));
            // ??
            var temp = a.Column(j).Scale((int) R[i,j]);
            a.Column(i).VecSub(temp);
        }
    }

    return new Tuple<Matrix, Matrix>(new Matrix(2,2), R);
}