function sumVectors(a, b) {
    return a.map((value, index) => value + b[index]);
}

function normBackwardRow(dy, x, gamma, eps = 1e-5) {
    const d = x.length;

    const mean = x.reduce((a, b) => a + b, 0) / d;
    const centered = x.map((v) => v - mean);
    const variance = centered.reduce((sum, v) => sum + v * v, 0) / d;
    const std = Math.sqrt(variance + eps);

    const xHat = centered.map((v) => v / std);
    const dGamma = dy.map((value, index) => value * xHat[index]);
    const dBeta = [...dy];
    const dXHat = dy.map((value, index) => value * gamma[index]);
    const meanDXHat = dXHat.reduce((a, b) => a + b, 0) / d;
    const meanDXHatXHat =
        dXHat.reduce((sum, value, i) => sum + value * xHat[i], 0) / d;

    return {
        dx: dXHat.map(
            (value, index) =>
                (value - meanDXHat - xHat[index] * meanDXHatXHat) / std,
        ),
        dGamma,
        dBeta,
    };
}

function sumRowGradients(rows) {
    if (rows.length === 0) {
        return [];
    }

    return rows.reduce((acc, row) => sumVectors(acc, row));
}

function ffnAddNormBackward(dOutput, addNorm2Input, gamma) {
    const rowGrads = dOutput.map((dyRow, rowIndex) =>
        normBackwardRow(dyRow, addNorm2Input[rowIndex], gamma),
    );
    const dAddNorm2Input = rowGrads.map((row) => row.dx);
    const dGamma = sumRowGradients(rowGrads.map((row) => row.dGamma));
    const dBeta = sumRowGradients(rowGrads.map((row) => row.dBeta));

    return {
        dAddNormOutput: dAddNorm2Input,
        dFfnOutput: dAddNorm2Input,
        grads: {
            dGamma,
            dBeta,
        },
    };
}

function addNormBackward(dOutput, addNorm1Input, gamma) {
    const rowGrads = dOutput.map((dyRow, rowIndex) =>
        normBackwardRow(dyRow, addNorm1Input[rowIndex], gamma),
    );
    const dAddNorm1Input = rowGrads.map((row) => row.dx);
    const dGamma = sumRowGradients(rowGrads.map((row) => row.dGamma));
    const dBeta = sumRowGradients(rowGrads.map((row) => row.dBeta));

    return {
        dPositionEmbeddings: dAddNorm1Input,
        dAttentionOutput: dAddNorm1Input,
        grads: {
            dGamma,
            dBeta,
        },
    };
}

export { ffnAddNormBackward, addNormBackward };
