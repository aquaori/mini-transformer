import { D_MODEL, VOCAB_SIZE } from "./config.js";
import fs from "fs";

const W_vocab = Array.from({ length: D_MODEL }, () =>
    Array.from({ length: VOCAB_SIZE }, () => (Math.random() * 2 - 1) * 0.02),
);

const b_vocab = Array.from({ length: VOCAB_SIZE }, () => 0);

// 将 W_vocab 和 b_vocab 保存到 vocab_head.js 文件中
const vocabHeadData = {
    W_vocab,
    b_vocab,
};
const fileContent = `// vocab_head.js
    // 该文件由 initVocabHead() 自动生成
    // 不要手动修改，除非你明确知道自己在做什么
    export const W_vocab = ${JSON.stringify(W_vocab, null, 2)};
    export const b_vocab = ${JSON.stringify(b_vocab, null, 2)};
    `;
fs.writeFileSync("./vocab_head.js", fileContent, "utf8");
console.log("vocab_head.js 已生成");
