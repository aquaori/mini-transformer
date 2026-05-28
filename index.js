import { generateLogits, generateMaskToken, mlm } from "./mlm.js";
import { getPositionalEmbeddings } from "./position.js";
import { calculateAttention } from "./attention.js";
import { add, norm } from "./add&norm.js";
import { tokenize } from "./tokenizer.js";
import { embed } from "./embedding.js";
import { ffn } from "./ffn.js";
import { Wo } from "./data.js";
import fs from "fs";

function Encoder(
    inputText,
    Wq,
    Wk,
    Wv,
    W1,
    b1,
    W2,
    b2,
    headNums,
    W_vocab,
    b_vocab,
) {
    const tokenIds = tokenize(inputText);

    // 加一个MLM占位符，随机替换一个 token 为 MASK_TOKEN_ID
    const [maskedTokenIds, maskIndex, originalTokenId] =
        generateMaskToken(tokenIds);

    const X = embed(maskedTokenIds);

    // 计算位置编码
    const positionEmbeddings = getPositionalEmbeddings(X, X[0].length);

    // 计算注意力输出
    const attentionOutput = calculateAttention(
        positionEmbeddings,
        Wq,
        Wk,
        Wv,
        headNums,
        Wo,
    );

    // Add&Norm
    const addNormOutput = attentionOutput.map((row, index) =>
        norm(add(positionEmbeddings[index], row)),
    );

    // FFN 前馈神经网络
    const ffnOutput = ffn(addNormOutput, W1, b1, W2, b2);

    // Add&Norm
    const ffnAddNormOutput = ffnOutput.map((row, index) =>
        norm(add(addNormOutput[index], row)),
    );

    // 处理完成后，计算logits，并进行MLM预测
    const logits = generateLogits(ffnAddNormOutput, W_vocab, b_vocab);
    const probabilities = mlm(logits, maskIndex);
    const hiddenAtMask = ffnAddNormOutput[maskIndex];

    return [probabilities, logits, maskIndex, originalTokenId, hiddenAtMask];
}

export { Encoder };
