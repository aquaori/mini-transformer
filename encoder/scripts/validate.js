import { crossEntropyLoss } from "../backward/backward.js";
import { forwardCache } from "../cache/forward-cache.js";
import { validationSets } from "../data/dataSets.js";
import { Encoder } from "../index.js";
import { detokenize, specialTokens, tokenIdToToken } from "../../shared/tokenizer.js";
import {
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
} from "../params/data.js";
import { W_vocab, b_vocab } from "../params/vocab_head.js";

function buildMaskedPreview(maskedTokenIds) {
    const previewTokenIds = maskedTokenIds.filter(
        (tokenId) =>
            tokenId !== specialTokens.BOS_TOKEN_ID &&
            tokenId !== specialTokens.EOS_TOKEN_ID,
    );

    return detokenize(previewTokenIds, {
        skipSpecialTokens: false,
    }).replaceAll(specialTokens.MASK_TOKEN, "[MASK]");
}

const inputText =
    validationSets[Math.floor(Math.random() * validationSets.length)];

const { probabilities } = await Encoder(
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
);

const { maskedTokenIds, originalTokenIds } = forwardCache.get("input");
const predictedTokenIds = probabilities.map((row) =>
    row.indexOf(Math.max(...row)),
);
const predictedProbabilities = probabilities.map(
    (row, index) => row[predictedTokenIds[index]],
);
const targetProbabilities = probabilities.map(
    (row, index) => row[originalTokenIds[index]],
);
const loss = crossEntropyLoss(probabilities, originalTokenIds);

console.log(`验证样本: ${inputText}`);
console.log(`Mask 后输入: ${buildMaskedPreview(maskedTokenIds)}`);
console.log(
    `原词: ${originalTokenIds.map((tokenId) => tokenIdToToken(tokenId)).join(" + ")} (ids=${originalTokenIds.join(",")})`,
);
console.log(
    `预测: ${predictedTokenIds.map((tokenId) => tokenIdToToken(tokenId)).join(" + ")} (ids=${predictedTokenIds.join(",")})`,
);
console.log(
    `平均最高概率: ${(predictedProbabilities.reduce((sum, value) => sum + value, 0) / predictedProbabilities.length).toFixed(4)}`,
);
console.log(
    `平均正确 token 概率: ${(targetProbabilities.reduce((sum, value) => sum + value, 0) / targetProbabilities.length).toFixed(4)}`,
);
console.log(
    `是否全猜对: ${predictedTokenIds.every((tokenId, index) => tokenId === originalTokenIds[index]) ? "是" : "否"}`,
);
console.log(`Loss: ${loss.toFixed(4)}`);
