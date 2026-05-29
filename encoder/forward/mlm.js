import { forwardCache } from "../cache/forward-cache.js";
import { matMul, softmaxRows } from "../../shared/utils.js";
import { specialTokens, isMaskableTokenId } from "../../shared/tokenizer.js";

async function generateLogits(hidden, W_vocab, b_vocab) {
    const logits = (await matMul(hidden, W_vocab)).map((row) =>
        row.map((value, index) => value + b_vocab[index]),
    );
    forwardCache.set("vocabHead", {
        hidden,
        W_vocab,
        b_vocab,
        logits,
    });

    return logits;
}

async function generateMaskedLogits(hiddenAtMask, W_vocab, b_vocab) {
    const logits = (await matMul(hiddenAtMask, W_vocab)).map((row) =>
        row.map((value, index) => value + b_vocab[index]),
    );

    forwardCache.set("vocabHead", {
        hiddenAtMask,
        W_vocab,
        b_vocab,
        logits,
    });

    return logits;
}

function generateMaskToken(tokenIds) {
    return generateWholeWordMaskToken(
        tokenIds,
        tokenIds
            .map((tokenId, index) => ({ text: null, indices: [index] }))
            .filter(({ indices: [index] }) => isMaskableTokenId(tokenIds[index])),
    );
}

function generateWholeWordMaskToken(tokenIds, wordSpans) {
    const maskedTokenIds = [...tokenIds];
    const maskableSpans = wordSpans.filter(({ indices }) =>
        indices.every((index) => isMaskableTokenId(tokenIds[index])),
    );

    if (maskableSpans.length === 0) {
        throw new Error("当前输入没有可用于 MLM 的普通 token");
    }

    const selectedSpan =
        maskableSpans[Math.floor(Math.random() * maskableSpans.length)];
    const maskIndices = [...selectedSpan.indices];
    const originalTokenIds = maskIndices.map((index) => maskedTokenIds[index]);

    for (const maskIndex of maskIndices) {
        maskedTokenIds[maskIndex] = specialTokens.MASK_TOKEN_ID;
    }

    forwardCache.set("mask", {
        tokenIds,
        maskedTokenIds,
        maskIndices,
        originalTokenIds,
        maskedText: selectedSpan.text,
    });

    return [maskedTokenIds, maskIndices, originalTokenIds];
}

function softmaxVector(values) {
    return softmaxRows([values])[0];
}

function mlm(logits, maskIndex) {
    // 这是mlm打分和softmax的主要部分，输入是模型最后输出的logits和mask token的位置，输出是一个完整的词表概率分布
    // 先打分，然后对打分结果进行softmax，得到概率分布
    const maskedTokenLogits = logits[maskIndex];
    const probabilities = softmaxVector(maskedTokenLogits);
    forwardCache.set("softmax", {
        logits,
        maskIndex,
        maskedTokenLogits,
        probabilities,
    });

    return probabilities;
}

function mlmFromMaskedLogits(maskedTokenLogits) {
    const probabilities = maskedTokenLogits.map((row) => softmaxVector(row));
    forwardCache.set("softmax", {
        maskedTokenLogits,
        probabilities,
    });

    return probabilities;
}

export {
    generateLogits,
    generateMaskedLogits,
    generateMaskToken,
    generateWholeWordMaskToken,
    mlm,
    mlmFromMaskedLogits,
};
