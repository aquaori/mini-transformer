// MLM 相关的损失函数和反向传播函数

function crossEntropyLoss(probs, logits, maskIndex, targetTokenId) {
    const loss = -Math.log(probs[targetTokenId] + 1e-12);
    return loss;
}

function crossEntropyBackward(probs, targetTokenId) {
    const dlogits = [...probs];
    dlogits[targetTokenId] -= 1;

    return dlogits;
}

export { crossEntropyLoss, crossEntropyBackward };
