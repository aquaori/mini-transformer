import fs from "fs";

import {
    TOKENIZER_TYPE,
    TOKENIZER_MIN_FREQUENCY,
    BPE_TARGET_VOCAB_SIZE,
    BPE_MIN_PAIR_FREQUENCY,
} from "../config/config.js";
import { trainSets, validationSets } from "../data/dataSets.js";
import {
    normalizeEnglishText,
    isMostlyEnglishText,
    splitWordTokens,
} from "../../shared/tokenizers/word-tokenizer-core.js";

const TOKENIZER_VOCAB_PATH = new URL(
    "../params/tokenizer-vocab.js",
    import.meta.url,
);

const SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<mask>"];

function collectPretokenFrequencies() {
    const frequencies = new Map();

    for (const text of [...trainSets, ...validationSets]) {
        const normalizedText = normalizeEnglishText(text);

        if (!normalizedText || !isMostlyEnglishText(normalizedText)) {
            continue;
        }

        for (const token of splitWordTokens(normalizedText)) {
            frequencies.set(token, (frequencies.get(token) ?? 0) + 1);
        }
    }

    return frequencies;
}

function buildWordVocabulary(pretokenFrequencies) {
    const learnedTokens = [...pretokenFrequencies.entries()]
        .filter(([, count]) => count >= TOKENIZER_MIN_FREQUENCY)
        .sort((left, right) => {
            if (right[1] !== left[1]) {
                return right[1] - left[1];
            }

            return left[0].localeCompare(right[0]);
        })
        .map(([token]) => token);

    return {
        meta: {
            type: "word",
            minFrequency: TOKENIZER_MIN_FREQUENCY,
        },
        tokens: [...SPECIAL_TOKENS, ...learnedTokens],
        merges: [],
    };
}

function tokenizeWordForBpe(word) {
    const chars = [...word];

    if (chars.length === 0) {
        return [];
    }

    chars[chars.length - 1] = `${chars[chars.length - 1]}</w>`;
    return chars;
}

function countPairs(wordEntries) {
    const pairCounts = new Map();

    for (const { symbols, frequency } of wordEntries) {
        for (let index = 0; index < symbols.length - 1; index++) {
            const pairKey = `${symbols[index]} ${symbols[index + 1]}`;
            pairCounts.set(pairKey, (pairCounts.get(pairKey) ?? 0) + frequency);
        }
    }

    return pairCounts;
}

function mergePairInSymbols(symbols, left, right) {
    const merged = [];

    for (let index = 0; index < symbols.length; index++) {
        const current = symbols[index];
        const next = symbols[index + 1];

        if (current === left && next === right) {
            merged.push(left + right);
            index++;
            continue;
        }

        merged.push(current);
    }

    return merged;
}

function buildBpeVocabulary(pretokenFrequencies) {
    const filteredPretokens = [...pretokenFrequencies.entries()].filter(
        ([, frequency]) => frequency >= TOKENIZER_MIN_FREQUENCY,
    );
    const wordEntries = filteredPretokens.map(
        ([word, frequency]) => ({
            word,
            frequency,
            symbols: tokenizeWordForBpe(word),
        }),
    );
    const symbolSet = new Set(
        wordEntries.flatMap(({ symbols }) => symbols),
    );
    const merges = [];

    while (SPECIAL_TOKENS.length + symbolSet.size < BPE_TARGET_VOCAB_SIZE) {
        const pairCounts = countPairs(wordEntries);
        let bestPairKey = null;
        let bestPairCount = 0;

        for (const [pairKey, count] of pairCounts.entries()) {
            if (
                count > bestPairCount ||
                (count === bestPairCount &&
                    bestPairKey !== null &&
                    pairKey.localeCompare(bestPairKey) < 0)
            ) {
                bestPairKey = pairKey;
                bestPairCount = count;
            }
        }

        if (
            !bestPairKey ||
            bestPairCount < BPE_MIN_PAIR_FREQUENCY ||
            merges.length > BPE_TARGET_VOCAB_SIZE * 4
        ) {
            break;
        }

        const [left, right] = bestPairKey.split(" ");
        const mergedToken = left + right;
        merges.push([left, right]);
        symbolSet.add(mergedToken);

        for (const entry of wordEntries) {
            entry.symbols = mergePairInSymbols(entry.symbols, left, right);
        }
    }

    return {
        meta: {
            type: "bpe",
            minFrequency: TOKENIZER_MIN_FREQUENCY,
            targetVocabSize: BPE_TARGET_VOCAB_SIZE,
            minPairFrequency: BPE_MIN_PAIR_FREQUENCY,
        },
        tokens: [
            ...SPECIAL_TOKENS,
            ...[...symbolSet].sort((left, right) => left.localeCompare(right)),
        ],
        merges,
    };
}

function serializeTokenizerVocab({ meta, tokens, merges }) {
    return `// tokenizer-vocab.js
// 该文件由 build-tokenizer-vocab.js 自动生成
// 不要手动修改，除非你明确知道自己在做什么

export const TOKENIZER_META = ${JSON.stringify(meta, null, 2)};

export const PAD_TOKEN = "<pad>";
export const UNK_TOKEN = "<unk>";
export const BOS_TOKEN = "<bos>";
export const EOS_TOKEN = "<eos>";
export const MASK_TOKEN = "<mask>";

export const PAD_TOKEN_ID = 0;
export const UNK_TOKEN_ID = 1;
export const BOS_TOKEN_ID = 2;
export const EOS_TOKEN_ID = 3;
export const MASK_TOKEN_ID = 4;

export const TOKEN_LIST = ${JSON.stringify(tokens, null, 2)};
export const TOKEN_TO_ID = Object.freeze(
  Object.fromEntries(TOKEN_LIST.map((token, index) => [token, index]))
);
export const BPE_MERGES = ${JSON.stringify(merges, null, 2)};
export const VOCAB_SIZE = TOKEN_LIST.length;
`;
}

function buildTokenizerVocabulary() {
    const pretokenFrequencies = collectPretokenFrequencies();

    if (TOKENIZER_TYPE === "word") {
        return buildWordVocabulary(pretokenFrequencies);
    }

    if (TOKENIZER_TYPE === "bpe") {
        return buildBpeVocabulary(pretokenFrequencies);
    }

    throw new Error(`不支持的 tokenizer 类型: ${TOKENIZER_TYPE}`);
}

const tokenizerVocab = buildTokenizerVocabulary();

fs.writeFileSync(
    TOKENIZER_VOCAB_PATH,
    serializeTokenizerVocab(tokenizerVocab),
    "utf8",
);

console.log("tokenizer-vocab.js 已生成");
console.log(`tokenizerType = ${tokenizerVocab.meta.type}`);
console.log(`vocabSize = ${tokenizerVocab.tokens.length}`);
if (tokenizerVocab.meta.type === "bpe") {
    console.log(`bpeMerges = ${tokenizerVocab.merges.length}`);
}
