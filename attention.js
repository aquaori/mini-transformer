// 注意力计算
import { matMul } from "./utils.js";

// 矩阵转置
function transpose(matrix) {
    return matrix[0].map((_, colIndex) => matrix.map((row) => row[colIndex]));
}

// 矩阵整体乘一个标量
function scaleMatrix(matrix, scale) {
    return matrix.map((row) => row.map((value) => value * scale));
}

// 对每一行做 softmax
function softmax(matrix) {
    return matrix.map((row) => {
        const max = Math.max(...row);
        const exps = row.map((x) => Math.exp(x - max));
        const sum = exps.reduce((a, b) => a + b, 0);

        return exps.map((x) => x / sum);
    });
}

// 拆分多头
function splitHeads(matrix, numHeads) {
    const dModel = matrix[0].length;

    if (dModel % numHeads !== 0) {
        throw new Error(`dModel=${dModel} 不能被 numHeads=${numHeads} 整除`);
    }

    const headDim = dModel / numHeads;

    return Array.from({ length: numHeads }, (_, headIndex) =>
        matrix.map((row) =>
            row.slice(headIndex * headDim, (headIndex + 1) * headDim),
        ),
    );
}

// 拼接多头
function concatHeads(heads) {
    const seqLen = heads[0].length;

    return Array.from({ length: seqLen }, (_, tokenIndex) =>
        heads.flatMap((head) => head[tokenIndex]),
    );
}

// 单头注意力
function singleHeadAttention(Q, K, V) {
    const dK = Q[0].length;

    // scores = QK^T / sqrt(dK)
    const scores = scaleMatrix(matMul(Q, transpose(K)), 1 / Math.sqrt(dK));

    // weights = softmax(scores)
    const weights = softmax(scores);

    // output = weightsV
    return matMul(weights, V);
}

// 多头注意力
function calculateAttention(
    positionEmbeddings,
    Wq,
    Wk,
    Wv,
    numHeads = 1,
    Wo = null,
) {
    // 1. 计算 Q / K / V
    const Q = matMul(positionEmbeddings, Wq);
    const K = matMul(positionEmbeddings, Wk);
    const V = matMul(positionEmbeddings, Wv);

    // 2. 拆成多个 head
    const QHeads = splitHeads(Q, numHeads);
    const KHeads = splitHeads(K, numHeads);
    const VHeads = splitHeads(V, numHeads);

    // 3. 每个 head 单独计算 attention
    const headOutputs = QHeads.map((QHead, index) =>
        singleHeadAttention(QHead, KHeads[index], VHeads[index]),
    );

    // 4. 拼接所有 head
    const concatOutput = concatHeads(headOutputs);

    // 5. 可选输出投影
    return Wo ? matMul(concatOutput, Wo) : concatOutput;
}

export { calculateAttention };
