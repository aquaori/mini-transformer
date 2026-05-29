import {
    generateMaskedLogits,
    generateWholeWordMaskToken,
    mlmFromMaskedLogits,
} from "./forward/mlm.js";
import { getPositionalEmbeddings } from "./forward/position.js";
import { calculateAttention } from "./forward/attention.js";
import { add, norm } from "./forward/add-norm.js";
import { forwardCache } from "./cache/forward-cache.js";
import { tokenizeWithMetadata } from "../shared/tokenizer.js";
import { embed } from "./forward/embedding.js";
import { ffn } from "./forward/ffn.js";

async function Encoder(
    inputText,
    Wq,
    Wk,
    Wv,
    W1,
    b1,
    W2,
    b2,
    Wo,
    gamma,
    beta,
    headNums,
    W_vocab,
    b_vocab,
) {
    forwardCache.reset();

    const { tokenIds, wordSpans } = tokenizeWithMetadata(inputText);

    // whole-word masking：同一个词对应的多个 piece 一起 mask
    const [maskedTokenIds, maskIndices, originalTokenIds] =
        generateWholeWordMaskToken(tokenIds, wordSpans);
    forwardCache.set("input", {
        inputText,
        tokenIds,
        wordSpans,
        maskedTokenIds,
        maskIndices,
        originalTokenIds,
    });

    const X = embed(maskedTokenIds);
    forwardCache.set("embedding", {
        tokenIds: maskedTokenIds,
        output: X,
    });

    const positionEmbeddings = getPositionalEmbeddings(X, X[0].length);
    const layerCaches = [];
    let hidden = positionEmbeddings;

    for (let layerIndex = 0; layerIndex < Wq.length; layerIndex++) {
        const attentionResult = await calculateAttention(
            hidden,
            Wq[layerIndex],
            Wk[layerIndex],
            Wv[layerIndex],
            headNums,
            Wo[layerIndex],
        );
        const attentionOutput = attentionResult.output;

        const addNorm1Input = attentionOutput.map((row, index) =>
            add(hidden[index], row),
        );
        const addNormOutput = addNorm1Input.map((row) =>
            norm(row, gamma[layerIndex], beta[layerIndex]),
        );

        const ffnResult = await ffn(
            addNormOutput,
            W1[layerIndex],
            b1[layerIndex],
            W2[layerIndex],
            b2[layerIndex],
        );
        const ffnOutput = ffnResult.output;

        const addNorm2Input = ffnOutput.map((row, index) =>
            add(addNormOutput[index], row),
        );
        const layerOutput = addNorm2Input.map((row) =>
            norm(row, gamma[layerIndex], beta[layerIndex]),
        );

        layerCaches.push({
            attention: attentionResult.cache,
            addNorm1: {
                residualInput: hidden,
                subLayerOutput: attentionOutput,
                normInput: addNorm1Input,
                gamma: gamma[layerIndex],
                beta: beta[layerIndex],
                output: addNormOutput,
            },
            ffn: ffnResult.cache,
            addNorm2: {
                residualInput: addNormOutput,
                subLayerOutput: ffnOutput,
                normInput: addNorm2Input,
                gamma: gamma[layerIndex],
                beta: beta[layerIndex],
                output: layerOutput,
            },
        });

        hidden = layerOutput;
    }

    forwardCache.set("position", {
        input: X,
        output: positionEmbeddings,
    });
    forwardCache.set("layers", layerCaches);

    const hiddenAtMask = maskIndices.map((maskIndex) => hidden[maskIndex]);
    // 处理完成后，只计算 mask 位置的 logits，并进行 MLM 预测
    const logits = await generateMaskedLogits(hiddenAtMask, W_vocab, b_vocab);
    const probabilities = mlmFromMaskedLogits(logits);
    forwardCache.set("mlm", {
        hidden,
        hiddenAtMask,
        logits,
        probabilities,
        maskIndices,
        originalTokenIds,
    });

    return { probabilities, logits };
}

export { Encoder };
