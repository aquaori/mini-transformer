// 矩阵乘法
function matMul(A, B) {
    const rowsA = A.length;
    const colsA = A[0].length;
    const rowsB = B.length;
    const colsB = B[0].length;

    if (colsA !== rowsB) {
        throw new Error(
            `矩阵维度不匹配: ${rowsA}x${colsA} 不能乘 ${rowsB}x${colsB}`,
        );
    }

    const result = [];

    for (let i = 0; i < rowsA; i++) {
        const row = [];

        for (let j = 0; j < colsB; j++) {
            let sum = 0;

            for (let k = 0; k < colsA; k++) {
                sum += A[i][k] * B[k][j];
            }

            row.push(sum);
        }

        result.push(row);
    }

    return result;
}

// 给矩阵的每一行加偏置
function addBias(matrix, bias) {
    return matrix.map((row) => row.map((value, index) => value + bias[index]));
}

export { matMul, addBias };
