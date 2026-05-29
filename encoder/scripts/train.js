import fs from "fs";
import {
    buildParamFileContents,
} from "../backward/paramsUpdate.js";
import { EMBEDDING_META } from "../params/embedding-table.js";
import { createInitialParams, trainModel } from "./training-core.js";
import { createConsoleStatusRenderer } from "./training-status.js";

const TRAINING_STATE_PATH = new URL(
    "../params/training_state.json",
    import.meta.url,
);
const DATA_PATH = new URL("../params/data.js", import.meta.url);
const VOCAB_HEAD_PATH = new URL("../params/vocab_head.js", import.meta.url);
const EMBEDDING_TABLE_PATH = new URL(
    "../params/embedding-table.js",
    import.meta.url,
);

function readNumberArg(name) {
    const prefix = `--${name}=`;
    const arg = process.argv.find((value) => value.startsWith(prefix));

    if (!arg) {
        return undefined;
    }

    const value = Number(arg.slice(prefix.length));
    return Number.isFinite(value) ? value : undefined;
}

function hasArg(name) {
    return process.argv.includes(`--${name}`);
}

function saveBestParams(params) {
    const fileContents = buildParamFileContents(params, EMBEDDING_META);

    fs.writeFileSync(DATA_PATH, fileContents.dataJs, "utf8");
    fs.writeFileSync(VOCAB_HEAD_PATH, fileContents.vocabHeadJs, "utf8");
    fs.writeFileSync(
        EMBEDDING_TABLE_PATH,
        fileContents.embeddingTableJs,
        "utf8",
    );
}

function loadBestValidationLoss() {
    if (!fs.existsSync(TRAINING_STATE_PATH)) {
        return Infinity;
    }

    try {
        const state = JSON.parse(fs.readFileSync(TRAINING_STATE_PATH, "utf8"));

        return Number.isFinite(state.bestValidationLoss)
            ? state.bestValidationLoss
            : Infinity;
    } catch {
        return Infinity;
    }
}

function saveTrainingState(bestValidationLoss) {
    if (!Number.isFinite(bestValidationLoss)) {
        return;
    }

    const state = {
        bestValidationLoss,
        updatedAt: new Date().toISOString(),
        runner: "node",
    };

    fs.writeFileSync(TRAINING_STATE_PATH, JSON.stringify(state, null, 2));
}

const renderStatus = createConsoleStatusRenderer();
const bestValidationLoss = loadBestValidationLoss();

if (!Number.isFinite(bestValidationLoss)) {
    console.log("没有找到之前的训练状态，使用默认参数重新训练。");
}

const result = await trainModel({
    initialParams: createInitialParams(),
    bestValidationLoss,
    maxRounds: readNumberArg("max-rounds"),
    batchSize: readNumberArg("batch-size"),
    validationInterval: readNumberArg("validation-interval"),
    validationSize: readNumberArg("validation-size"),
    reportStatus: async (status) => {
        renderStatus(status);
    },
    onBestParams: async (bestParams, nextBestValidationLoss) => {
        if (hasArg("dry-run")) {
            return;
        }
        saveBestParams(bestParams);
        saveTrainingState(nextBestValidationLoss);
    },
    onValidationComplete: async ({ currentParams, bestValidationLoss }) => {
        if (hasArg("dry-run")) {
            return;
        }
        saveBestParams(currentParams);
        saveTrainingState(bestValidationLoss);
    },
});

if (!hasArg("dry-run")) {
    saveBestParams(result.bestParams);
    saveTrainingState(result.bestValidationLoss);
}

if (process.stdout.isTTY) {
    process.stdout.write("\n");
}
console.log(
    `Node training finished. Best validation loss: ${result.bestValidationLoss.toFixed(4)}`,
);
