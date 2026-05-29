import { crossEntropyBackward, crossEntropyLoss } from "../backward/backward.js";
import { forwardCache } from "../cache/forward-cache.js";
import { Encoder } from "../index.js";
import { tokenIdToToken } from "../../shared/tokenizer.js";
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

const { probabilities, logits } = await Encoder(
    "A small model can learn simple patterns from repeated practice.",
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
const { originalTokenIds } = forwardCache.get("input");

const loss = crossEntropyLoss(probabilities, originalTokenIds);
crossEntropyBackward(probabilities, originalTokenIds);
const predictedTokenIds = probabilities.map((row) =>
    row.indexOf(Math.max(...row)),
);
const originalTokens = originalTokenIds.map((tokenId) => tokenIdToToken(tokenId));
const predictedTokens = predictedTokenIds.map((tokenId) => tokenIdToToken(tokenId));

console.log(
    `原始 token: ${originalTokens.join(" + ")} (ids=${originalTokenIds.join(",")}), ` +
        `预测 token: ${predictedTokens.join(" + ")} (ids=${predictedTokenIds.join(",")}), ` +
        `最高概率: ${Math.max(...probabilities.flat()).toFixed(4)}, ` +
        `平均正确 token 概率: ${(originalTokenIds.reduce((sum, tokenId, index) => sum + probabilities[index][tokenId], 0) / originalTokenIds.length).toFixed(4)}, ` +
        `损失: ${loss.toFixed(4)}`,
);
