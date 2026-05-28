import { Wq, Wk, Wv, W1, b1, W2, b2, headNums } from "./data.js";
import {
    MAX_TRAIN_ROUND,
    EARLY_STOP_PATIENCE,
    LEARNING_RATE,
    VALIDATION_INTERVAL,
} from "./config.js";
import { W_vocab, b_vocab } from "./vocab_head.js";
import { Encoder } from "./index.js";
import { crossEntropyLoss, crossEntropyBackward } from "./backward.js";
import fs from "fs";
import { testSets, validationSets } from "./dataSets.js";

function train() {
    let bestValidationLoss = Infinity;
    let badSteps = 0;
    let bestParams = { W_vocab, b_vocab };
    let currentParams = { W_vocab, b_vocab };

    for (let round = 1; round <= MAX_TRAIN_ROUND; round++) {
        const inputText = testSets[Math.floor(Math.random() * testSets.length)];
        const [
            probabilities,
            logits,
            maskIndex,
            originalTokenId,
            hiddenAtMask,
        ] = Encoder(
            inputText,
            Wq,
            Wk,
            Wv,
            W1,
            b1,
            W2,
            b2,
            headNums,
            currentParams.W_vocab,
            currentParams.b_vocab,
        );

        const loss = crossEntropyLoss(
            probabilities,
            logits,
            maskIndex,
            originalTokenId,
        );
        const dLogits = crossEntropyBackward(probabilities, originalTokenId);

        console.log(`Round ${round}: Loss = ${loss.toFixed(4)}`);
        currentParams = updateParameters(currentParams, hiddenAtMask, dLogits);

        if (round % VALIDATION_INTERVAL === 0) {
            const validationLosses = validationSets.map((text) => {
                const [
                    probabilities,
                    logits,
                    maskIndex,
                    originalTokenId,
                    hiddenAtMask,
                ] = Encoder(
                    text,
                    Wq,
                    Wk,
                    Wv,
                    W1,
                    b1,
                    W2,
                    b2,
                    headNums,
                    currentParams.W_vocab,
                    currentParams.b_vocab,
                );
                return crossEntropyLoss(
                    probabilities,
                    logits,
                    maskIndex,
                    originalTokenId,
                );
            });
            const avgValidationLoss =
                validationLosses.reduce((a, b) => a + b, 0) /
                validationLosses.length;
            console.log(
                `Validation ${round}: Loss = ${avgValidationLoss.toFixed(4)}`,
            );

            if (avgValidationLoss < bestValidationLoss) {
                bestValidationLoss = avgValidationLoss;
                badSteps = 0;
                bestParams = {
                    W_vocab: currentParams.W_vocab,
                    b_vocab: currentParams.b_vocab,
                };
            } else {
                badSteps++;
            }
        }
        if (badSteps >= EARLY_STOP_PATIENCE) {
            console.log(
                `验证损失已连续 ${badSteps} 次未下降，最佳验证损失： ${bestValidationLoss.toFixed(4)}，训练提前结束。`,
            );
            break;
        }
    }
    saveBestParams(bestParams);
    console.log(
        `训练结束，最佳验证损失： ${bestValidationLoss.toFixed(4)}，参数已更新。`,
    );
}

function updateParameters(params, hiddenAtMask, dLogits) {
    const newW_vocab = params.W_vocab.map((row, i) =>
        row.map(
            (value, j) => value - LEARNING_RATE * hiddenAtMask[i] * dLogits[j],
        ),
    );

    const newb_vocab = params.b_vocab.map(
        (value, j) => value - LEARNING_RATE * dLogits[j],
    );

    return { W_vocab: newW_vocab, b_vocab: newb_vocab };
}

function saveBestParams(params) {
    const fileContent = `// vocab_head.js
    // 该文件由 train() 自动生成
    // 不要手动修改，除非你明确知道自己在做什么
    export const W_vocab = ${JSON.stringify(params.W_vocab, null, 2)};
    export const b_vocab = ${JSON.stringify(params.b_vocab, null, 2)};
    `;
    fs.writeFileSync("./vocab_head.js", fileContent, "utf8");
}

train();
train();
train();
train();
train();
