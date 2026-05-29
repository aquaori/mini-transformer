import { backwardCache } from "../cache/backward-cache.js";
import { forwardCache } from "../cache/forward-cache.js";
import { ffnAddNormBackward } from "./ffnAddNorm.js";
import { vocabHeadBackward } from "./vocabHead.js";
import { ffnBackward } from "./ffn.js";
import { addNormBackward } from "./ffnAddNorm.js";
import { attentionBackward } from "./attention.js";
import { embeddingBackward } from "./embedding.js";
import { D_MODEL } from "../config/config.js";
import { VOCAB_SIZE } from "../params/tokenizer-vocab.js";

function crossEntropyLoss(probs, targetTokenId) {
    if (Array.isArray(targetTokenId)) {
        const losses = targetTokenId.map(
            (tokenId, index) => -Math.log(probs[index][tokenId] + 1e-12),
        );

        return losses.reduce((sum, loss) => sum + loss, 0) / losses.length;
    }

    return -Math.log(probs[targetTokenId] + 1e-12);
}

function crossEntropyBackward(probs, targetTokenId) {
    if (Array.isArray(targetTokenId)) {
        return probs.map((row, index) => {
            const dlogits = [...row];
            dlogits[targetTokenId[index]] -= 1;
            return dlogits;
        });
    }

    const dlogits = [...probs];
    dlogits[targetTokenId] -= 1;

    return dlogits;
}

function scatterMaskGradient(seqLen, dModel, maskIndices, dHiddenAtMask) {
    const grads = Array.from({ length: seqLen }, () =>
        new Array(dModel).fill(0),
    );

    for (let index = 0; index < maskIndices.length; index++) {
        grads[maskIndices[index]] = [...dHiddenAtMask[index]];
    }

    return grads;
}

function backward(probabilities, W_vocab) {
    // 负责从尾到头的反向传播，计算每一层的梯度，一路传播回去
    // 然后将中间所有的梯度保存在 backwardCache 中，最后再一次性更新参数
    backwardCache.reset();
    // 先计算损失函数
    const { originalTokenIds } = forwardCache.get("input");
    const loss = crossEntropyLoss(probabilities, originalTokenIds);
    backwardCache.set("loss", loss);

    // 计算输出层的梯度
    const dLogits = crossEntropyBackward(probabilities, originalTokenIds);
    backwardCache.set("dLogits", dLogits);

    // 计算 vocabHead 的梯度
    const { hiddenAtMask } = forwardCache.get("mlm");
    const vocabGrads = vocabHeadBackward(hiddenAtMask, W_vocab, dLogits);
    backwardCache.set("vocabHead", vocabGrads);

    // 将梯度映射回隐藏层
    const maskIndices = forwardCache.get("input").maskIndices;
    const seqLen = forwardCache.get("input").tokenIds.length;
    const grad = scatterMaskGradient(
        seqLen,
        D_MODEL,
        maskIndices,
        vocabGrads.dHiddenAtMask,
    );
    const layerCaches = forwardCache.get("layers");
    const layerBackwardStates = [];
    let currentGrad = grad;

    for (let layerIndex = layerCaches.length - 1; layerIndex >= 0; layerIndex--) {
        const layerCache = layerCaches[layerIndex];
        const addNorm2Grads = ffnAddNormBackward(
            currentGrad,
            layerCache.addNorm2.normInput,
            layerCache.addNorm2.gamma,
        );
        const ffnGrads = ffnBackward(addNorm2Grads.dFfnOutput, layerCache.ffn);
        const dAddNormOutputTotal = addNorm2Grads.dAddNormOutput.map((row, i) =>
            row.map((value, j) => value + ffnGrads.dx[i][j]),
        );
        const addNorm1Grads = addNormBackward(
            dAddNormOutputTotal,
            layerCache.addNorm1.normInput,
            layerCache.addNorm1.gamma,
        );
        const attentionGrads = attentionBackward(
            addNorm1Grads.dAttentionOutput,
            layerCache.attention,
        );
        currentGrad = addNorm1Grads.dPositionEmbeddings.map((row, i) =>
            row.map((value, j) => value + attentionGrads.dx[i][j]),
        );

        layerBackwardStates[layerIndex] = {
            addNorm2: {
                ...addNorm2Grads,
                dAddNormOutputTotal,
            },
            ffn: ffnGrads,
            addNorm1: addNorm1Grads,
            attention: attentionGrads,
        };
    }

    backwardCache.set("layers", layerBackwardStates);
    backwardCache.set("position", {
        dPositionEmbeddingsTotal: currentGrad,
    });

    // 计算 embedding 的梯度
    const { maskedTokenIds } = forwardCache.get("input");
    const embeddingGrads = embeddingBackward(
        currentGrad,
        maskedTokenIds,
        VOCAB_SIZE,
        D_MODEL,
    );

    backwardCache.set("embedding", embeddingGrads);

    return {
        loss,
        grads: backwardCache.all(),
    };
}

export {
    crossEntropyLoss,
    crossEntropyBackward,
    scatterMaskGradient,
    backward,
};
