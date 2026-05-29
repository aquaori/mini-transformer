import { LEARNING_RATE } from "../config/config.js";

function vocabHeadBackward(hiddenAtMask, W_vocab, dLogits) {
    const dW_vocab = W_vocab.map((row) => row.map(() => 0));
    const db_vocab = new Array(dLogits[0].length).fill(0);
    const dHiddenAtMask = hiddenAtMask.map((hiddenRow, rowIndex) =>
        hiddenRow.map((_, hiddenIndex) =>
            dLogits[rowIndex].reduce(
                (sum, dLogit, vocabIndex) =>
                    sum + W_vocab[hiddenIndex][vocabIndex] * dLogit,
                0,
            ),
        ),
    );

    for (let rowIndex = 0; rowIndex < hiddenAtMask.length; rowIndex++) {
        for (let hiddenIndex = 0; hiddenIndex < hiddenAtMask[rowIndex].length; hiddenIndex++) {
            for (let vocabIndex = 0; vocabIndex < dLogits[rowIndex].length; vocabIndex++) {
                dW_vocab[hiddenIndex][vocabIndex] +=
                    hiddenAtMask[rowIndex][hiddenIndex] *
                    dLogits[rowIndex][vocabIndex];
            }
        }

        for (let vocabIndex = 0; vocabIndex < dLogits[rowIndex].length; vocabIndex++) {
            db_vocab[vocabIndex] += dLogits[rowIndex][vocabIndex];
        }
    }

    return {
        dHiddenAtMask,
        grads: {
            dW_vocab,
            db_vocab,
        },
    };
}

function applyGradients(params, grads) {
    return {
        W_vocab: params.W_vocab.map((row, i) =>
            row.map((value, j) => value - LEARNING_RATE * grads.dW_vocab[i][j]),
        ),
        b_vocab: params.b_vocab.map(
            (value, j) => value - LEARNING_RATE * grads.db_vocab[j],
        ),
    };
}

export { vocabHeadBackward, applyGradients };
