// 注意力计算
import {
    matMul,
    transpose,
    scaleMatrix,
    softmaxRows,
} from "../../shared/utils.js";

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
async function singleHeadAttention(Q, K, V) {
    const dK = Q[0].length;

    // scores = QK^T / sqrt(dK)
    const scores = scaleMatrix(
        await matMul(Q, transpose(K)),
        1 / Math.sqrt(dK),
    );

    // weights = softmax(scores)
    const weights = softmaxRows(scores);

    // output = weightsV
    return {
        scores,
        weights,
        output: await matMul(weights, V),
    };
}

// 多头注意力
async function calculateAttention(
    positionEmbeddings,
    Wq,
    Wk,
    Wv,
    numHeads = 1,
    Wo = null,
) {
    // 1. 计算 Q / K / V
    const Q = await matMul(positionEmbeddings, Wq);
    const K = await matMul(positionEmbeddings, Wk);
    const V = await matMul(positionEmbeddings, Wv);

    // 2. 拆成多个 head
    const QHeads = splitHeads(Q, numHeads);
    const KHeads = splitHeads(K, numHeads);
    const VHeads = splitHeads(V, numHeads);

    // 3. 每个 head 单独计算 attention
    const headResults = await Promise.all(
        QHeads.map((QHead, index) =>
            singleHeadAttention(QHead, KHeads[index], VHeads[index]),
        ),
    );
    const headOutputs = headResults.map((head) => head.output);

    // 4. 拼接所有 head
    const concatOutput = concatHeads(headOutputs);

    // 5. 可选输出投影
    const output = Wo ? await matMul(concatOutput, Wo) : concatOutput;
    const cache = {
        input: positionEmbeddings,
        Wq,
        Wk,
        Wv,
        Wo,
        numHeads,
        Q,
        K,
        V,
        QHeads,
        KHeads,
        VHeads,
        headResults,
        concatOutput,
        output,
    };

    return { output, cache };
}

export { calculateAttention };
