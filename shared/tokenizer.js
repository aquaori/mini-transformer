import { TOKENIZER_TYPE } from "../encoder/config/config.js";
import { tokenizer as charTokenizer } from "./tokenizers/char-tokenizer.js";
import { tokenizer as bpeTokenizer } from "./tokenizers/bpe-tokenizer.js";
import { tokenizer as wordTokenizer } from "./tokenizers/word-tokenizer.js";

const tokenizer =
    TOKENIZER_TYPE === "char"
        ? charTokenizer
        : TOKENIZER_TYPE === "bpe"
          ? bpeTokenizer
          : wordTokenizer;
const {
    encode: tokenize,
    encodeWithMetadata: tokenizeWithMetadata,
    decode: detokenize,
    tokenIdToToken,
    tokenToId,
    isSpecialTokenId,
    isMaskableTokenId,
    getVocabSize,
    getSpecialTokens,
} = tokenizer;

const specialTokens = getSpecialTokens();

export {
    tokenizer,
    tokenize,
    tokenizeWithMetadata,
    detokenize,
    tokenIdToToken,
    tokenToId,
    isSpecialTokenId,
    isMaskableTokenId,
    getVocabSize,
    getSpecialTokens,
    specialTokens,
};
