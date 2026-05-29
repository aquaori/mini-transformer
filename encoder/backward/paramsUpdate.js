import { LEARNING_RATE } from "../config/config.js";

function cloneMatrix(matrix) {
    return matrix.map((row) => [...row]);
}

function cloneMatrixList(matrices) {
    return matrices.map((matrix) => cloneMatrix(matrix));
}

function cloneVector(vector) {
    return [...vector];
}

function cloneVectorList(vectors) {
    return vectors.map((vector) => cloneVector(vector));
}

function cloneFlatMatrix(flatMatrix) {
    return new Float32Array(flatMatrix);
}

function addMatrix(left, right) {
    return left.map((row, i) => row.map((value, j) => value + right[i][j]));
}

function addVector(left, right) {
    return left.map((value, index) => value + right[index]);
}

function scaleMatrix(matrix, factor) {
    return matrix.map((row) => row.map((value) => value * factor));
}

function scaleVector(vector, factor) {
    return vector.map((value) => value * factor);
}

function updateMatrix(matrix, gradMatrix, learningRate = LEARNING_RATE) {
    return matrix.map((row, i) =>
        row.map((value, j) => value - learningRate * gradMatrix[i][j]),
    );
}

function updateVector(vector, gradVector, learningRate = LEARNING_RATE) {
    return vector.map(
        (value, index) => value - learningRate * gradVector[index],
    );
}

function updateFlatMatrix(
    flatMatrix,
    gradMatrix,
    rows,
    cols,
    learningRate = LEARNING_RATE,
) {
    const next = new Float32Array(flatMatrix.length);

    for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
            const index = row * cols + col;
            next[index] = flatMatrix[index] - learningRate * gradMatrix[row][col];
        }
    }

    return next;
}

function applyAllGradients(params, backwardState, learningRate = LEARNING_RATE) {
    const vocabHeadGrads = backwardState.vocabHead?.grads;
    const embeddingGrads = backwardState.embedding?.grads;
    const layerStates = backwardState.layers ?? [];

    return {
        ...params,
        W_vocab: vocabHeadGrads
            ? updateMatrix(params.W_vocab, vocabHeadGrads.dW_vocab, learningRate)
            : params.W_vocab,
        b_vocab: vocabHeadGrads
            ? updateVector(params.b_vocab, vocabHeadGrads.db_vocab, learningRate)
            : params.b_vocab,
        W1: layerStates.length
            ? params.W1.map((matrix, index) =>
                  updateMatrix(
                      matrix,
                      layerStates[index].ffn.grads.dW1,
                      learningRate,
                  ),
              )
            : params.W1,
        b1: layerStates.length
            ? params.b1.map((vector, index) =>
                  updateVector(
                      vector,
                      layerStates[index].ffn.grads.db1,
                      learningRate,
                  ),
              )
            : params.b1,
        W2: layerStates.length
            ? params.W2.map((matrix, index) =>
                  updateMatrix(
                      matrix,
                      layerStates[index].ffn.grads.dW2,
                      learningRate,
                  ),
              )
            : params.W2,
        b2: layerStates.length
            ? params.b2.map((vector, index) =>
                  updateVector(
                      vector,
                      layerStates[index].ffn.grads.db2,
                      learningRate,
                  ),
              )
            : params.b2,
        Wq: layerStates.length
            ? params.Wq.map((matrix, index) =>
                  updateMatrix(
                      matrix,
                      layerStates[index].attention.grads.dWq,
                      learningRate,
                  ),
              )
            : params.Wq,
        Wk: layerStates.length
            ? params.Wk.map((matrix, index) =>
                  updateMatrix(
                      matrix,
                      layerStates[index].attention.grads.dWk,
                      learningRate,
                  ),
              )
            : params.Wk,
        Wv: layerStates.length
            ? params.Wv.map((matrix, index) =>
                  updateMatrix(
                      matrix,
                      layerStates[index].attention.grads.dWv,
                      learningRate,
                  ),
              )
            : params.Wv,
        Wo: layerStates.length
            ? params.Wo.map((matrix, index) =>
                  updateMatrix(
                      matrix,
                      layerStates[index].attention.grads.dWo,
                      learningRate,
                  ),
              )
            : params.Wo,
        gamma: layerStates.length
            ? params.gamma.map((vector, index) =>
                  updateVector(
                      vector,
                      addVector(
                          layerStates[index].addNorm1.grads.dGamma,
                          layerStates[index].addNorm2.grads.dGamma,
                      ),
                      learningRate,
                  ),
              )
            : params.gamma,
        beta: layerStates.length
            ? params.beta.map((vector, index) =>
                  updateVector(
                      vector,
                      addVector(
                          layerStates[index].addNorm1.grads.dBeta,
                          layerStates[index].addNorm2.grads.dBeta,
                      ),
                      learningRate,
                  ),
              )
            : params.beta,
        embeddingMatrix: embeddingGrads
            ? updateFlatMatrix(
                  params.embeddingMatrix,
                  embeddingGrads.dEmbeddingTable,
                  embeddingGrads.dEmbeddingTable.length,
                  embeddingGrads.dEmbeddingTable[0].length,
                  learningRate,
              )
            : params.embeddingMatrix,
    };
}

function cloneModelParams(params) {
    return {
        Wq: cloneMatrixList(params.Wq),
        Wk: cloneMatrixList(params.Wk),
        Wv: cloneMatrixList(params.Wv),
        W1: cloneMatrixList(params.W1),
        b1: cloneVectorList(params.b1),
        W2: cloneMatrixList(params.W2),
        b2: cloneVectorList(params.b2),
        Wo: cloneMatrixList(params.Wo),
        gamma: cloneVectorList(params.gamma),
        beta: cloneVectorList(params.beta),
        headNums: params.headNums,
        W_vocab: cloneMatrix(params.W_vocab),
        b_vocab: cloneVector(params.b_vocab),
        embeddingMatrix: cloneFlatMatrix(params.embeddingMatrix),
    };
}

function averageBackwardStates(backwardStates) {
    if (backwardStates.length === 0) {
        throw new Error("backwardStates 不能为空");
    }

    const factor = 1 / backwardStates.length;
    const first = backwardStates[0];
    const summed = {
        vocabHead: {
            grads: {
                dW_vocab: cloneMatrix(first.vocabHead.grads.dW_vocab),
                db_vocab: cloneVector(first.vocabHead.grads.db_vocab),
            },
        },
        layers: first.layers.map((layer) => ({
            ffn: {
                grads: {
                    dW1: cloneMatrix(layer.ffn.grads.dW1),
                    db1: cloneVector(layer.ffn.grads.db1),
                    dW2: cloneMatrix(layer.ffn.grads.dW2),
                    db2: cloneVector(layer.ffn.grads.db2),
                },
            },
            attention: {
                grads: {
                    dWq: cloneMatrix(layer.attention.grads.dWq),
                    dWk: cloneMatrix(layer.attention.grads.dWk),
                    dWv: cloneMatrix(layer.attention.grads.dWv),
                    dWo: cloneMatrix(layer.attention.grads.dWo),
                },
            },
            addNorm1: {
                grads: {
                    dGamma: cloneVector(layer.addNorm1.grads.dGamma),
                    dBeta: cloneVector(layer.addNorm1.grads.dBeta),
                },
            },
            addNorm2: {
                grads: {
                    dGamma: cloneVector(layer.addNorm2.grads.dGamma),
                    dBeta: cloneVector(layer.addNorm2.grads.dBeta),
                },
            },
        })),
        embedding: {
            grads: {
                dEmbeddingTable: cloneMatrix(
                    first.embedding.grads.dEmbeddingTable,
                ),
            },
        },
    };

    for (let index = 1; index < backwardStates.length; index++) {
        const current = backwardStates[index];

        summed.vocabHead.grads.dW_vocab = addMatrix(
            summed.vocabHead.grads.dW_vocab,
            current.vocabHead.grads.dW_vocab,
        );
        summed.vocabHead.grads.db_vocab = addVector(
            summed.vocabHead.grads.db_vocab,
            current.vocabHead.grads.db_vocab,
        );

        for (let layerIndex = 0; layerIndex < summed.layers.length; layerIndex++) {
            summed.layers[layerIndex].ffn.grads.dW1 = addMatrix(
                summed.layers[layerIndex].ffn.grads.dW1,
                current.layers[layerIndex].ffn.grads.dW1,
            );
            summed.layers[layerIndex].ffn.grads.db1 = addVector(
                summed.layers[layerIndex].ffn.grads.db1,
                current.layers[layerIndex].ffn.grads.db1,
            );
            summed.layers[layerIndex].ffn.grads.dW2 = addMatrix(
                summed.layers[layerIndex].ffn.grads.dW2,
                current.layers[layerIndex].ffn.grads.dW2,
            );
            summed.layers[layerIndex].ffn.grads.db2 = addVector(
                summed.layers[layerIndex].ffn.grads.db2,
                current.layers[layerIndex].ffn.grads.db2,
            );

            summed.layers[layerIndex].attention.grads.dWq = addMatrix(
                summed.layers[layerIndex].attention.grads.dWq,
                current.layers[layerIndex].attention.grads.dWq,
            );
            summed.layers[layerIndex].attention.grads.dWk = addMatrix(
                summed.layers[layerIndex].attention.grads.dWk,
                current.layers[layerIndex].attention.grads.dWk,
            );
            summed.layers[layerIndex].attention.grads.dWv = addMatrix(
                summed.layers[layerIndex].attention.grads.dWv,
                current.layers[layerIndex].attention.grads.dWv,
            );
            summed.layers[layerIndex].attention.grads.dWo = addMatrix(
                summed.layers[layerIndex].attention.grads.dWo,
                current.layers[layerIndex].attention.grads.dWo,
            );

            summed.layers[layerIndex].addNorm1.grads.dGamma = addVector(
                summed.layers[layerIndex].addNorm1.grads.dGamma,
                current.layers[layerIndex].addNorm1.grads.dGamma,
            );
            summed.layers[layerIndex].addNorm1.grads.dBeta = addVector(
                summed.layers[layerIndex].addNorm1.grads.dBeta,
                current.layers[layerIndex].addNorm1.grads.dBeta,
            );

            summed.layers[layerIndex].addNorm2.grads.dGamma = addVector(
                summed.layers[layerIndex].addNorm2.grads.dGamma,
                current.layers[layerIndex].addNorm2.grads.dGamma,
            );
            summed.layers[layerIndex].addNorm2.grads.dBeta = addVector(
                summed.layers[layerIndex].addNorm2.grads.dBeta,
                current.layers[layerIndex].addNorm2.grads.dBeta,
            );
        }

        summed.embedding.grads.dEmbeddingTable = addMatrix(
            summed.embedding.grads.dEmbeddingTable,
            current.embedding.grads.dEmbeddingTable,
        );
    }

    return {
        vocabHead: {
            grads: {
                dW_vocab: scaleMatrix(summed.vocabHead.grads.dW_vocab, factor),
                db_vocab: scaleVector(summed.vocabHead.grads.db_vocab, factor),
            },
        },
        layers: summed.layers.map((layer) => ({
            ffn: {
                grads: {
                    dW1: scaleMatrix(layer.ffn.grads.dW1, factor),
                    db1: scaleVector(layer.ffn.grads.db1, factor),
                    dW2: scaleMatrix(layer.ffn.grads.dW2, factor),
                    db2: scaleVector(layer.ffn.grads.db2, factor),
                },
            },
            attention: {
                grads: {
                    dWq: scaleMatrix(layer.attention.grads.dWq, factor),
                    dWk: scaleMatrix(layer.attention.grads.dWk, factor),
                    dWv: scaleMatrix(layer.attention.grads.dWv, factor),
                    dWo: scaleMatrix(layer.attention.grads.dWo, factor),
                },
            },
            addNorm1: {
                grads: {
                    dGamma: scaleVector(layer.addNorm1.grads.dGamma, factor),
                    dBeta: scaleVector(layer.addNorm1.grads.dBeta, factor),
                },
            },
            addNorm2: {
                grads: {
                    dGamma: scaleVector(layer.addNorm2.grads.dGamma, factor),
                    dBeta: scaleVector(layer.addNorm2.grads.dBeta, factor),
                },
            },
        })),
        embedding: {
            grads: {
                dEmbeddingTable: scaleMatrix(
                    summed.embedding.grads.dEmbeddingTable,
                    factor,
                ),
            },
        },
    };
}

function serializeDataParams(params) {
    return `// data.js
// 该文件由训练流程自动生成
// 不要手动修改，除非你明确知道自己在做什么

const Wq = ${JSON.stringify(params.Wq, null, 4)};
const Wk = ${JSON.stringify(params.Wk, null, 4)};
const Wv = ${JSON.stringify(params.Wv, null, 4)};
const W1 = ${JSON.stringify(params.W1, null, 4)};
const b1 = ${JSON.stringify(params.b1, null, 4)};
const W2 = ${JSON.stringify(params.W2, null, 4)};
const b2 = ${JSON.stringify(params.b2, null, 4)};
const Wo = ${JSON.stringify(params.Wo, null, 4)};
const gamma = ${JSON.stringify(params.gamma, null, 4)};
const beta = ${JSON.stringify(params.beta, null, 4)};
const headNums = ${JSON.stringify(params.headNums)};

export { Wq, Wk, Wv, W1, b1, W2, b2, Wo, gamma, beta, headNums };
`;
}

function serializeVocabHeadParams(params) {
    return `// vocab_head.js
// 该文件由训练流程自动生成
// 不要手动修改，除非你明确知道自己在做什么

export const W_vocab = ${JSON.stringify(params.W_vocab, null, 2)};
export const b_vocab = ${JSON.stringify(params.b_vocab, null, 2)};
`;
}

function serializeEmbeddingParams(params, embeddingMeta) {
    return `// embedding-table.js
// 该文件由训练流程自动生成
// 不要手动修改，除非你明确知道自己在做什么

export const EMBEDDING_META = ${JSON.stringify(embeddingMeta, null, 2)};

export const embeddingMatrix = new Float32Array(${JSON.stringify(
        Array.from(params.embeddingMatrix),
        null,
        2,
    )});
`;
}

function buildParamFileContents(params, embeddingMeta) {
    return {
        dataJs: serializeDataParams(params),
        vocabHeadJs: serializeVocabHeadParams(params),
        embeddingTableJs: serializeEmbeddingParams(params, embeddingMeta),
    };
}

export {
    updateMatrix,
    updateVector,
    updateFlatMatrix,
    applyAllGradients,
    cloneModelParams,
    averageBackwardStates,
    buildParamFileContents,
    serializeDataParams,
    serializeVocabHeadParams,
    serializeEmbeddingParams,
};
