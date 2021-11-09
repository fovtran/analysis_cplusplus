int det(int matrixSize, int matrix[][matrixSize]){
    int determinant = 0, matrixValues[matrixSize * matrixSize], matrixFirstRowValues[matrixSize * matrixSize];
    for(int i = matrixSize; i > 2; i--){
        for(int row = 0; row < matrixSize; row++){
          for(int col = 0; col < matrixSize; col++){
            matrixFirstRowValues[row + (matrixSize - i)] = matrix[1][col + (matrixSize - i)]; 
            //copies the first row values for an array until we have a 2x2 matrix
          }
        } 
    }

    //multiply the values in matrix Values by their respective matrix without 
    //the row and column of these values times -1^row+col

    determinant += (matrix[matrixSize-1][matrixSize-1] * matrix[matrixSize-2][matrixSize-2])
                 - (matrix[matrixSize-1][matrixSize-2] * matrix[matrixSize-2][matrixSize-1]);
    return determinant;
}