import { addMatrices, matMulCpu, transpose } from "../../shared/utils.js";

function softmaxBackwardRow(dWeightsRow, weightsRow) {
    const dot = dWeightsRow.reduce(
        (sum, value, index) => sum + value * weightsRow[index],
        0,
    );

    return weightsRow.map(
        (weight, index) => weight * (dWeightsRow[index] - dot),
    );
}

function softmaxBackward(dWeights, weights) {
    return dWeights.map((row, index) =>
        softmaxBackwardRow(row, weights[index]),
    );
}

function concatHeadsBackwardStyle(heads) {
    const seqLen = heads[0].length;

    return Array.from({ length: seqLen }, (_, tokenIndex) =>
        heads.flatMap((head) => head[tokenIndex]),
    );
}

function singleHeadAttentionBackward(dHeadOutput, cache) {
    const { Q, K, V, weights } = cache;
    const dKScale = Math.sqrt(Q[0].length);

    // output = weights × V
    const dWeights = matMulCpu(dHeadOutput, transpose(V));
    const dV = matMulCpu(transpose(weights), dHeadOutput);

    // weights = softmax(scores)
    const dScores = softmaxBackward(dWeights, weights);

    // scores = (Q × K^T) / sqrt(dk)
    const dScaledScores = dScores.map((row) =>
        row.map((value) => value / dKScale),
    );

    const dQ = matMulCpu(dScaledScores, K);
    const dK = matMulCpu(transpose(dScaledScores), Q);

    return {
        dQ,
        dK,
        dV,
    };
}

function splitConcatGradient(dConcatOutput, numHeads) {
    const dModel = dConcatOutput[0].length;
    const headDim = dModel / numHeads;

    return Array.from({ length: numHeads }, (_, headIndex) =>
        dConcatOutput.map((row) =>
            row.slice(headIndex * headDim, (headIndex + 1) * headDim),
        ),
    );
}

function attentionBackward(dAttentionOutput, cache) {
    const {
        input,
        Wq,
        Wk,
        Wv,
        Wo,
        QHeads,
        KHeads,
        VHeads,
        headResults,
        concatOutput,
        numHeads,
    } = cache;

    // output = concatOutput × Wo
    const dWo = matMulCpu(transpose(concatOutput), dAttentionOutput);
    const dConcatOutput = matMulCpu(dAttentionOutput, transpose(Wo));

    // 拆回每个 head
    const dHeadOutputs = splitConcatGradient(dConcatOutput, numHeads);

    const headBackwardResults = dHeadOutputs.map((dHeadOutput, i) =>
        singleHeadAttentionBackward(dHeadOutput, {
            Q: QHeads[i],
            K: KHeads[i],
            V: VHeads[i],
            weights: headResults[i].weights,
        }),
    );

    const dQHeads = headBackwardResults.map((result) => result.dQ);
    const dKHeads = headBackwardResults.map((result) => result.dK);
    const dVHeads = headBackwardResults.map((result) => result.dV);

    // 把多个 head 拼回完整 dModel
    const dQ = concatHeadsBackwardStyle(dQHeads);
    const dK = concatHeadsBackwardStyle(dKHeads);
    const dV = concatHeadsBackwardStyle(dVHeads);

    // Q = input × Wq
    // K = input × Wk
    // V = input × Wv
    const dWq = matMulCpu(transpose(input), dQ);
    const dWk = matMulCpu(transpose(input), dK);
    const dWv = matMulCpu(transpose(input), dV);

    const dInputFromQ = matMulCpu(dQ, transpose(Wq));
    const dInputFromK = matMulCpu(dK, transpose(Wk));
    const dInputFromV = matMulCpu(dV, transpose(Wv));

    const dx = addMatrices(addMatrices(dInputFromQ, dInputFromK), dInputFromV);

    return {
        dx,
        grads: {
            dWq,
            dWk,
            dWv,
            dWo,
        },
    };
}

export { attentionBackward };
