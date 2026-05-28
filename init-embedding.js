import fs from "fs";

import { VOCAB_SIZE, D_MODEL, EMBEDDING_SCALE } from "./config.js";

// 简单固定种子的伪随机数生成器
function createRandom(seed = 42) {
    let value = seed >>> 0;

    return function random() {
        value += 0x6d2b79f5;

        let t = value;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);

        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function createEmbeddingMatrix() {
    const random = createRandom(42);
    const values = [];

    for (let tokenId = 0; tokenId < VOCAB_SIZE; tokenId++) {
        for (let dim = 0; dim < D_MODEL; dim++) {
            const value = (random() * 2 - 1) * EMBEDDING_SCALE;
            values.push(Number(value.toFixed(8)));
        }
    }

    return values;
}

const values = createEmbeddingMatrix();

const fileContent = `// embedding-table.js
// 该文件由 init-embedding.js 自动生成
// 不要手动修改，除非你明确知道自己在做什么

export const EMBEDDING_META = {
  vocabSize: ${VOCAB_SIZE},
  dModel: ${D_MODEL},
};

export const embeddingMatrix = new Float32Array([
${values
    .map((value, index) => {
        const suffix = index === values.length - 1 ? "" : ",";
        return `  ${value}${suffix}`;
    })
    .join("\n")}
]);
`;

fs.writeFileSync("./embedding-table.js", fileContent, "utf8");

console.log("embedding-table.js 已生成");
console.log(`vocabSize = ${VOCAB_SIZE}`);
console.log(`dModel = ${D_MODEL}`);
console.log(`参数量 = ${VOCAB_SIZE * D_MODEL}`);
