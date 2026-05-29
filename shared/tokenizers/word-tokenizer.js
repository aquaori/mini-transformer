import {
    TOKEN_TO_ID,
    TOKEN_LIST,
    VOCAB_SIZE,
    PAD_TOKEN,
    PAD_TOKEN_ID,
    UNK_TOKEN,
    UNK_TOKEN_ID,
    BOS_TOKEN,
    BOS_TOKEN_ID,
    EOS_TOKEN,
    EOS_TOKEN_ID,
    MASK_TOKEN,
    MASK_TOKEN_ID,
} from "../../encoder/params/tokenizer-vocab.js";
import { splitWordTokens, joinWordTokens } from "./word-tokenizer-core.js";

const SPECIAL_TOKEN_IDS = new Set([
    PAD_TOKEN_ID,
    UNK_TOKEN_ID,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    MASK_TOKEN_ID,
]);

function encodeWithMetadata(text, options = {}) {
    const { addBos = true, addEos = true } = options;
    const tokenIds = [];
    const wordSpans = [];

    if (addBos) {
        tokenIds.push(BOS_TOKEN_ID);
    }

    for (const token of splitWordTokens(text)) {
        const start = tokenIds.length;
        tokenIds.push(TOKEN_TO_ID[token] ?? UNK_TOKEN_ID);
        wordSpans.push({
            text: token,
            indices: [start],
        });
    }

    if (addEos) {
        tokenIds.push(EOS_TOKEN_ID);
    }

    return { tokenIds, wordSpans };
}

function encode(text, options = {}) {
    return encodeWithMetadata(text, options).tokenIds;
}

function decode(tokenIds, options = {}) {
    const { skipSpecialTokens = true } = options;
    const tokens = [];

    for (const tokenId of tokenIds) {
        const token = TOKEN_LIST[tokenId];

        if (token === undefined) {
            throw new Error(`无法解码非法 token id: ${tokenId}`);
        }

        if (SPECIAL_TOKEN_IDS.has(tokenId)) {
            if (skipSpecialTokens) {
                continue;
            }
        }

        tokens.push(token);
    }

    return joinWordTokens(tokens);
}

function tokenIdToToken(tokenId) {
    return TOKEN_LIST[tokenId] ?? `<invalid:${tokenId}>`;
}

function tokenToId(token) {
    return TOKEN_TO_ID[token] ?? UNK_TOKEN_ID;
}

function isSpecialTokenId(tokenId) {
    return SPECIAL_TOKEN_IDS.has(tokenId);
}

function isMaskableTokenId(tokenId) {
    return !SPECIAL_TOKEN_IDS.has(tokenId);
}

function getVocabSize() {
    return VOCAB_SIZE;
}

function getSpecialTokens() {
    return {
        PAD_TOKEN,
        PAD_TOKEN_ID,
        UNK_TOKEN,
        UNK_TOKEN_ID,
        BOS_TOKEN,
        BOS_TOKEN_ID,
        EOS_TOKEN,
        EOS_TOKEN_ID,
        MASK_TOKEN,
        MASK_TOKEN_ID,
    };
}

const tokenizer = {
    type: "word",
    encode,
    encodeWithMetadata,
    decode,
    tokenIdToToken,
    tokenToId,
    isSpecialTokenId,
    isMaskableTokenId,
    getVocabSize,
    getSpecialTokens,
};

export { tokenizer };
