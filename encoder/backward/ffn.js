import { matMulCpu, transpose, sumRows } from "../../shared/utils.js";

function reluBackward(dActivated, hidden) {
    return dActivated.map((row, i) =>
        row.map((value, j) => (hidden[i][j] > 0 ? value : 0)),
    );
}

function ffnBackward(dOutput, cache) {
    const { input, W1, W2, hidden, activated } = cache;

    // output = activated × W2 + b2
    const dW2 = matMulCpu(transpose(activated), dOutput);
    const db2 = sumRows(dOutput);
    const dActivated = matMulCpu(dOutput, transpose(W2));

    // activated = relu(hidden)
    const dHidden = reluBackward(dActivated, hidden);

    // hidden = input × W1 + b1
    const dW1 = matMulCpu(transpose(input), dHidden);
    const db1 = sumRows(dHidden);
    const dx = matMulCpu(dHidden, transpose(W1));

    return {
        dx,
        grads: {
            dW1,
            db1,
            dW2,
            db2,
        },
    };
}

export { ffnBackward };
