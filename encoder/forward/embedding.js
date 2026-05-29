import { D_MODEL } from "../config/config.js";
import { VOCAB_SIZE } from "../params/tokenizer-vocab.js";

import { EMBEDDING_META, embeddingMatrix } from "../params/embedding-table.js";

if (EMBEDDING_META.vocabSize !== VOCAB_SIZE) {
    throw new Error(
        `embedding vocabSize 不匹配: ${EMBEDDING_META.vocabSize} !== ${VOCAB_SIZE}`,
    );
}

if (EMBEDDING_META.dModel !== D_MODEL) {
    throw new Error(
        `embedding dModel 不匹配: ${EMBEDDING_META.dModel} !== ${D_MODEL}`,
    );
}

function getEmbedding(tokenId) {
    if (!Number.isInteger(tokenId)) {
        throw new Error(`tokenId 必须是整数: ${tokenId}`);
    }

    if (tokenId < 0 || tokenId >= VOCAB_SIZE) {
        throw new Error(`tokenId 越界: ${tokenId}`);
    }

    const start = tokenId * D_MODEL;
    const end = start + D_MODEL;

    return Array.from(embeddingMatrix.slice(start, end));
}

function embed(tokenIds) {
    return tokenIds.map((tokenId) => getEmbedding(tokenId));
}

export { embeddingMatrix, getEmbedding, embed };
