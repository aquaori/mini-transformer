import fs from "fs";

import {
    D_FF,
    D_MODEL,
    HEAD_NUMS,
    NUM_LAYERS,
    PARAM_INIT_SCALE,
} from "../config/config.js";
import { serializeDataParams } from "../backward/paramsUpdate.js";

const DATA_PATH = new URL("../params/data.js", import.meta.url);

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

function createMatrix(rows, cols, random) {
    return Array.from({ length: rows }, () =>
        Array.from({ length: cols }, () =>
            Number(((random() * 2 - 1) * PARAM_INIT_SCALE).toFixed(8)),
        ),
    );
}

function createVector(length, value = 0) {
    return Array.from({ length }, () => value);
}

function createIdentity(size) {
    return Array.from({ length: size }, (_, row) =>
        Array.from({ length: size }, (_, col) => (row === col ? 1 : 0)),
    );
}

function validateConfig() {
    if (D_MODEL % HEAD_NUMS !== 0) {
        throw new Error(
            `D_MODEL=${D_MODEL} 不能被 HEAD_NUMS=${HEAD_NUMS} 整除`,
        );
    }

    if (D_FF <= 0) {
        throw new Error(`D_FF 必须大于 0: ${D_FF}`);
    }

    if (NUM_LAYERS <= 0) {
        throw new Error(`NUM_LAYERS 必须大于 0: ${NUM_LAYERS}`);
    }
}

function createParams() {
    validateConfig();

    const random = createRandom();

    return {
        Wq: Array.from({ length: NUM_LAYERS }, () =>
            createMatrix(D_MODEL, D_MODEL, random),
        ),
        Wk: Array.from({ length: NUM_LAYERS }, () =>
            createMatrix(D_MODEL, D_MODEL, random),
        ),
        Wv: Array.from({ length: NUM_LAYERS }, () =>
            createMatrix(D_MODEL, D_MODEL, random),
        ),
        W1: Array.from({ length: NUM_LAYERS }, () =>
            createMatrix(D_MODEL, D_FF, random),
        ),
        b1: Array.from({ length: NUM_LAYERS }, () => createVector(D_FF)),
        W2: Array.from({ length: NUM_LAYERS }, () =>
            createMatrix(D_FF, D_MODEL, random),
        ),
        b2: Array.from({ length: NUM_LAYERS }, () => createVector(D_MODEL)),
        Wo: Array.from({ length: NUM_LAYERS }, () => createIdentity(D_MODEL)),
        gamma: Array.from({ length: NUM_LAYERS }, () =>
            createVector(D_MODEL, 1),
        ),
        beta: Array.from({ length: NUM_LAYERS }, () => createVector(D_MODEL)),
        headNums: HEAD_NUMS,
    };
}

function writeDataFile(params) {
    fs.writeFileSync(DATA_PATH, serializeDataParams(params), "utf8");
}

const params = createParams();

writeDataFile(params);

console.log("data.js 已生成");
console.log(`dModel = ${D_MODEL}`);
console.log(`dFF = ${D_FF}`);
console.log(`numLayers = ${NUM_LAYERS}`);
console.log(`headNums = ${HEAD_NUMS}`);
