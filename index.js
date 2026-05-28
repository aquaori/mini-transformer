import { Wq, Wk, Wv, W1, b1, W2, b2, Wo, headNums } from "./data.js";
import { getPositionalEmbeddings } from "./position.js";
import { tokenize, detokenize } from "./tokenizer.js";
import { calculateAttention } from "./attention.js";
import { add, norm } from "./add&norm.js";
import { embed } from "./embedding.js";
import { ffn } from "./ffn.js";
import fs from "fs";

function Encoder(Wq, Wk, Wv, W1, b1, W2, b2, headNums) {
    // 输入文本
    const inputText = "Hello, Mini Attention!";
    const tokenIds = tokenize(inputText);
    const X = embed(tokenIds);

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

    // 输出结果到json文件中
    const outputData = {
        positionEmbeddings,
        attentionOutput,
        addNormOutput,
        ffnOutput,
        ffnAddNormOutput,
    };
    fs.writeFileSync("output.json", JSON.stringify(outputData, null, 2));
    console.log("计算完成，结果已保存到output.json文件中。");
}

Encoder(Wq, Wk, Wv, W1, b1, W2, b2, headNums);
