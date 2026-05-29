import {
    TOKENIZER_META,
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
    BPE_MERGES,
} from "../../encoder/params/tokenizer-vocab.js";
import { splitWordTokens, joinWordTokens } from "./word-tokenizer-core.js";

const END_OF_WORD = "</w>";
const SPECIAL_TOKEN_IDS = new Set([
    PAD_TOKEN_ID,
    UNK_TOKEN_ID,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    MASK_TOKEN_ID,
]);
const mergeRanks = new Map(
    BPE_MERGES.map(([left, right], index) => [`${left} ${right}`, index]),
);

function tokenizePretoken(pretoken) {
    const chars = [...pretoken];

    if (chars.length === 0) {
        return [];
    }

    chars[chars.length - 1] = `${chars[chars.length - 1]}${END_OF_WORD}`;
    return chars;
}

function findBestPair(symbols) {
    let bestPairIndex = -1;
    let bestPairRank = Infinity;

    for (let index = 0; index < symbols.length - 1; index++) {
        const pairKey = `${symbols[index]} ${symbols[index + 1]}`;
        const rank = mergeRanks.get(pairKey);

        if (rank === undefined) {
            continue;
        }

        if (rank < bestPairRank) {
            bestPairRank = rank;
            bestPairIndex = index;
        }
    }

    return bestPairIndex;
}

function applyBpe(pretoken) {
    let symbols = tokenizePretoken(pretoken);

    while (symbols.length > 1) {
        const bestPairIndex = findBestPair(symbols);

        if (bestPairIndex < 0) {
            break;
        }

        const merged = symbols[bestPairIndex] + symbols[bestPairIndex + 1];
        symbols = [
            ...symbols.slice(0, bestPairIndex),
            merged,
            ...symbols.slice(bestPairIndex + 2),
        ];
    }

    return symbols;
}

function encodeWithMetadata(text, options = {}) {
    const { addBos = true, addEos = true } = options;
    const tokenIds = [];
    const wordSpans = [];

    if (addBos) {
        tokenIds.push(BOS_TOKEN_ID);
    }

    for (const pretoken of splitWordTokens(text)) {
        const indices = [];

        for (const symbol of applyBpe(pretoken)) {
            const index = tokenIds.length;
            tokenIds.push(TOKEN_TO_ID[symbol] ?? UNK_TOKEN_ID);
            indices.push(index);
        }

        if (indices.length > 0) {
            wordSpans.push({
                text: pretoken,
                indices,
            });
        }
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
    const pretokens = [];
    let currentPretoken = "";

    for (const tokenId of tokenIds) {
        const token = TOKEN_LIST[tokenId];

        if (token === undefined) {
            throw new Error(`无法解码非法 token id: ${tokenId}`);
        }

        if (SPECIAL_TOKEN_IDS.has(tokenId)) {
            if (skipSpecialTokens) {
                continue;
            }

            if (currentPretoken) {
                pretokens.push(currentPretoken);
                currentPretoken = "";
            }

            pretokens.push(token);
            continue;
        }

        if (token.endsWith(END_OF_WORD)) {
            currentPretoken += token.slice(0, -END_OF_WORD.length);
            pretokens.push(currentPretoken);
            currentPretoken = "";
            continue;
        }

        currentPretoken += token;
    }

    if (currentPretoken) {
        pretokens.push(currentPretoken);
    }

    const plainPretokens = pretokens.filter(
        (token) =>
            token !== PAD_TOKEN &&
            token !== UNK_TOKEN &&
            token !== BOS_TOKEN &&
            token !== EOS_TOKEN &&
            token !== MASK_TOKEN,
    );

    if (!skipSpecialTokens) {
        return joinWordTokens(pretokens);
    }

    return joinWordTokens(plainPretokens);
}

function formatDisplayToken(token) {
    return token.endsWith(END_OF_WORD)
        ? token.slice(0, -END_OF_WORD.length)
        : token;
}

function tokenIdToToken(tokenId) {
    const token = TOKEN_LIST[tokenId];
    return token === undefined ? `<invalid:${tokenId}>` : formatDisplayToken(token);
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
    type: TOKENIZER_META.type,
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
