import { D_MODEL, VOCAB_SIZE } from "./config.js";
import { matMul } from "./utils.js";

function generateLogits(hidden, W_vocab, b_vocab) {
    return matMul(hidden, W_vocab).map((row) =>
        row.map((value, index) => value + b_vocab[index]),
    );
}

function generateMaskToken(tokenIds) {
    const maskedTokenIds = [...tokenIds];
    let maskIndex = Math.floor(Math.random() * tokenIds.length);
    // 不能 mask <bos> 和 <eos>，所以如果随机选中了它们，就继续随机选择，直到选中一个普通 token
    while (
        maskedTokenIds[maskIndex] === 256 || // BOS_TOKEN_ID
        maskedTokenIds[maskIndex] === 257 // EOS_TOKEN_ID
    ) {
        maskIndex = Math.floor(Math.random() * tokenIds.length);
    }
    const originalTokenId = maskedTokenIds[maskIndex];
    maskedTokenIds[maskIndex] = 258; // 使用 MASK_TOKEN_ID 替换选中的 token
    return [maskedTokenIds, maskIndex, originalTokenId];
}

function softmaxVector(values) {
    const max = Math.max(...values);
    const exps = values.map((x) => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);

    return exps.map((x) => x / sum);
}

function mlm(logits, maskIndex) {
    // 这是mlm打分和softmax的主要部分，输入是模型最后输出的logits和mask token的位置，输出是一个完整的词表概率分布
    // 先打分，然后对打分结果进行softmax，得到概率分布
    const maskedTokenLogits = logits[maskIndex];
    const probabilities = softmaxVector(maskedTokenLogits);
    return probabilities;
}

export { generateLogits, generateMaskToken, mlm };
