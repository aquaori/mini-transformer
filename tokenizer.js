import { BOS_TOKEN_ID, EOS_TOKEN_ID } from "./config.js";

const encoder = new TextEncoder();
const decoder = new TextDecoder("utf-8");

function tokenize(str, options = {}) {
    const { addBos = true, addEos = true } = options;

    const bytes = Array.from(encoder.encode(str));

    const tokenIds = [];

    if (addBos) {
        tokenIds.push(BOS_TOKEN_ID);
    }

    tokenIds.push(...bytes);

    if (addEos) {
        tokenIds.push(EOS_TOKEN_ID);
    }

    return tokenIds;
}

function detokenize(tokenIds, options = {}) {
    const { skipSpecialTokens = true } = options;

    const bytes = [];

    for (const id of tokenIds) {
        if (id >= 0 && id <= 255) {
            bytes.push(id);
            continue;
        }

        if (skipSpecialTokens) {
            continue;
        }

        throw new Error(`无法解码特殊 token 或非法 token id: ${id}`);
    }

    return decoder.decode(new Uint8Array(bytes));
}

export { tokenize, detokenize };
