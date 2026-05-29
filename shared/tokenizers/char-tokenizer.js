const PAD_TOKEN = "<pad>";
const UNK_TOKEN = "<unk>";
const BOS_TOKEN = "<bos>";
const EOS_TOKEN = "<eos>";
const MASK_TOKEN = "<mask>";

const BYTE_VOCAB_SIZE = 256;
const PAD_TOKEN_ID = 256;
const UNK_TOKEN_ID = 257;
const BOS_TOKEN_ID = 258;
const EOS_TOKEN_ID = 259;
const MASK_TOKEN_ID = 260;

const encoder = new TextEncoder();
const decoder = new TextDecoder("utf-8");
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

    for (const byte of encoder.encode(text)) {
        const start = tokenIds.length;
        tokenIds.push(byte);
        wordSpans.push({
            text: String.fromCharCode(byte),
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
    const bytes = [];
    const tokens = [];

    for (const tokenId of tokenIds) {
        if (tokenId >= 0 && tokenId < BYTE_VOCAB_SIZE) {
            bytes.push(tokenId);
            continue;
        }

        if (SPECIAL_TOKEN_IDS.has(tokenId)) {
            if (!skipSpecialTokens) {
                tokens.push(tokenIdToToken(tokenId));
            }
            continue;
        }

        tokens.push(UNK_TOKEN);
    }

    const text = decoder.decode(new Uint8Array(bytes));
    return skipSpecialTokens ? text : `${tokens.join(" ")} ${text}`.trim();
}

function tokenIdToToken(tokenId) {
    if (tokenId >= 0 && tokenId < BYTE_VOCAB_SIZE) {
        return String.fromCharCode(tokenId);
    }

    switch (tokenId) {
        case PAD_TOKEN_ID:
            return PAD_TOKEN;
        case UNK_TOKEN_ID:
            return UNK_TOKEN;
        case BOS_TOKEN_ID:
            return BOS_TOKEN;
        case EOS_TOKEN_ID:
            return EOS_TOKEN;
        case MASK_TOKEN_ID:
            return MASK_TOKEN;
        default:
            return `<invalid:${tokenId}>`;
    }
}

function tokenToId(token) {
    if (token.length === 1) {
        return token.charCodeAt(0);
    }

    switch (token) {
        case PAD_TOKEN:
            return PAD_TOKEN_ID;
        case UNK_TOKEN:
            return UNK_TOKEN_ID;
        case BOS_TOKEN:
            return BOS_TOKEN_ID;
        case EOS_TOKEN:
            return EOS_TOKEN_ID;
        case MASK_TOKEN:
            return MASK_TOKEN_ID;
        default:
            return UNK_TOKEN_ID;
    }
}

function isSpecialTokenId(tokenId) {
    return SPECIAL_TOKEN_IDS.has(tokenId);
}

function isMaskableTokenId(tokenId) {
    return !SPECIAL_TOKEN_IDS.has(tokenId) && tokenId !== 32;
}

function getVocabSize() {
    return MASK_TOKEN_ID + 1;
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
    type: "char",
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
