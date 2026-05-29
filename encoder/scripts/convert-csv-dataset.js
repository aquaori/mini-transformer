import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import {
    normalizeEnglishText,
    isMostlyEnglishText,
} from "../../shared/tokenizers/word-tokenizer-core.js";

const DATA_DIR = new URL("../data/", import.meta.url);
const OUTPUT_PATH = new URL("../data/externalData.js", import.meta.url);
const DATA_DIR_PATH = fileURLToPath(DATA_DIR);

function readNumberArg(name, fallback) {
    const prefix = `--${name}=`;
    const arg = process.argv.find((value) => value.startsWith(prefix));

    if (!arg) {
        return fallback;
    }

    const value = Number(arg.slice(prefix.length));
    return Number.isFinite(value) ? value : fallback;
}

const limits = {
    train: readNumberArg("train-limit", 20000),
    validation: readNumberArg("validation-limit", 2000),
    test: readNumberArg("test-limit", 2000),
};

const minWords = readNumberArg("min-words", 5);
const maxWords = readNumberArg("max-words", 48);

function createRandom(seed = 20260529) {
    let value = seed >>> 0;

    return function random() {
        value += 0x6d2b79f5;

        let t = value;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);

        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

const random = createRandom();

function splitIntoSentences(text) {
    return text
        .replace(/""/g, '"')
        .replace(/\s+/g, " ")
        .split(/(?<=[.!?])\s+/)
        .map((sentence) => normalizeEnglishText(sentence))
        .filter(Boolean);
}

function isUsefulSentence(sentence) {
    if (!isMostlyEnglishText(sentence)) {
        return false;
    }

    const wordCount = sentence.split(/\s+/).filter(Boolean).length;

    return wordCount >= minWords && wordCount <= maxWords;
}

function reservoirPush(buffer, limit, value, seenCount) {
    if (limit <= 0) {
        return seenCount;
    }

    const nextSeenCount = seenCount + 1;

    if (buffer.length < limit) {
        buffer.push(value);
        return nextSeenCount;
    }

    const replaceIndex = Math.floor(random() * nextSeenCount);

    if (replaceIndex < limit) {
        buffer[replaceIndex] = value;
    }

    return nextSeenCount;
}

async function parseSingleColumnCsv(filePath, onRow) {
    const stream = fs.createReadStream(filePath, { encoding: "utf8" });

    let field = "";
    let row = [];
    let inQuotes = false;

    for await (const chunk of stream) {
        for (let index = 0; index < chunk.length; index++) {
            const char = chunk[index];
            const next = chunk[index + 1];

            if (char === '"') {
                if (inQuotes && next === '"') {
                    field += '"';
                    index++;
                    continue;
                }

                inQuotes = !inQuotes;
                continue;
            }

            if (char === "," && !inQuotes) {
                row.push(field);
                field = "";
                continue;
            }

            if ((char === "\n" || char === "\r") && !inQuotes) {
                if (char === "\r" && next === "\n") {
                    index++;
                }

                row.push(field);
                field = "";

                if (row.length > 0) {
                    await onRow(row);
                }

                row = [];
                continue;
            }

            field += char;
        }
    }

    if (field.length > 0 || row.length > 0) {
        row.push(field);
        await onRow(row);
    }
}

async function collectSplitSamples(filePaths, limit) {
    const samples = [];
    let seenCount = 0;

    for (const filePath of filePaths) {
        let isHeader = true;

        await parseSingleColumnCsv(filePath, async (row) => {
            if (isHeader) {
                isHeader = false;
                return;
            }

            const [text = ""] = row;

            for (const sentence of splitIntoSentences(text)) {
                if (!isUsefulSentence(sentence)) {
                    continue;
                }

                seenCount = reservoirPush(samples, limit, sentence, seenCount);
            }
        });
    }

    return samples;
}

function getSplitFiles(splitPrefix) {
    return fs
        .readdirSync(DATA_DIR_PATH)
        .filter(
            (fileName) =>
                fileName.startsWith(splitPrefix) && fileName.endsWith(".csv"),
        )
        .sort()
        .map((fileName) => path.join(DATA_DIR_PATH, fileName));
}

function serializeDataFile(trainSets, validationSets, testSets) {
    return `// externalData.js
// 该文件由 convert-csv-dataset.js 自动生成
// 不要手动修改，除非你明确知道自己在做什么

const trainSets = ${JSON.stringify(trainSets, null, 2)};

const validationSets = ${JSON.stringify(validationSets, null, 2)};

const testSets = ${JSON.stringify(testSets, null, 2)};

export { trainSets, validationSets, testSets };
`;
}

const trainFiles = getSplitFiles("train-");
const validationFiles = getSplitFiles("validation-");
const testFiles = getSplitFiles("test-");

if (trainFiles.length === 0 || validationFiles.length === 0 || testFiles.length === 0) {
    throw new Error("未找到完整的 train / validation / test CSV 文件");
}

const trainSets = await collectSplitSamples(trainFiles, limits.train);
const validationSets = await collectSplitSamples(
    validationFiles,
    limits.validation,
);
const testSets = await collectSplitSamples(testFiles, limits.test);

fs.writeFileSync(
    OUTPUT_PATH,
    serializeDataFile(trainSets, validationSets, testSets),
    "utf8",
);

console.log("externalData.js 已生成");
console.log(`trainSets = ${trainSets.length}`);
console.log(`validationSets = ${validationSets.length}`);
console.log(`testSets = ${testSets.length}`);
console.log(`minWords = ${minWords}`);
console.log(`maxWords = ${maxWords}`);
