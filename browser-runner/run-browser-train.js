import fs from "fs";
import http from "http";
import path from "path";
import { fileURLToPath } from "url";
import { buildParamFileContents } from "../encoder/backward/paramsUpdate.js";
import { EMBEDDING_META } from "../encoder/params/embedding-table.js";
import { createConsoleStatusRenderer } from "../encoder/scripts/training-status.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(__dirname, "..");
const TRAINING_STATE_PATH = path.join(
    ROOT_DIR,
    "encoder",
    "params",
    "training_state.json",
);
const VOCAB_HEAD_PATH = path.join(
    ROOT_DIR,
    "encoder",
    "params",
    "vocab_head.js",
);
const DATA_PATH = path.join(ROOT_DIR, "encoder", "params", "data.js");
const EMBEDDING_TABLE_PATH = path.join(
    ROOT_DIR,
    "encoder",
    "params",
    "embedding-table.js",
);
const renderStatus = createConsoleStatusRenderer();

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

function readBestValidationLoss() {
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

function contentType(filePath) {
    if (filePath.endsWith(".html")) return "text/html";
    if (filePath.endsWith(".js")) return "text/javascript";
    if (filePath.endsWith(".json")) return "application/json";
    if (filePath.endsWith(".wasm")) return "application/wasm";

    return "application/octet-stream";
}

function createStaticServer() {
    const server = http.createServer((req, res) => {
        const url = new URL(req.url, "http://127.0.0.1");
        const requestedPath = decodeURIComponent(url.pathname);
        const filePath = path.normalize(
            path.join(
                ROOT_DIR,
                requestedPath === "/"
                    ? "browser-runner/index.html"
                    : requestedPath,
            ),
        );

        if (!filePath.startsWith(ROOT_DIR)) {
            res.writeHead(403);
            res.end("Forbidden");
            return;
        }

        fs.readFile(filePath, (error, content) => {
            if (error) {
                res.writeHead(404);
                res.end("Not found");
                return;
            }

            res.writeHead(200, { "Content-Type": contentType(filePath) });
            res.end(content);
        });
    });

    return new Promise((resolve) => {
        server.listen(0, "127.0.0.1", () => {
            const { port } = server.address();
            resolve({ server, port });
        });
    });
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

function saveTrainingState(bestValidationLoss) {
    if (!Number.isFinite(bestValidationLoss)) {
        return;
    }

    fs.writeFileSync(
        TRAINING_STATE_PATH,
        JSON.stringify(
            {
                bestValidationLoss,
                updatedAt: new Date().toISOString(),
                runner: "browser",
            },
            null,
            2,
        ),
    );
}

function persistTrainCheckpoint(checkpoint) {
    saveBestParams(checkpoint.currentParams);
    saveTrainingState(checkpoint.bestValidationLoss);
}

async function main() {
    let chromium;

    try {
        ({ chromium } = await import("playwright"));
    } catch {
        throw new Error(
            "Playwright is not installed. Run `npm install -D playwright` first.",
        );
    }

    const { server, port } = await createStaticServer();
    const browser = await chromium.launch({
        executablePath:
            "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        headless: true,
        args: ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist"],
    });

    try {
        const page = await browser.newPage();
        page.on("pageerror", (error) => console.error(error));
        await page.exposeFunction("reportTrainStatus", (status) => {
            renderStatus(status);
        });
        await page.exposeFunction("persistTrainCheckpoint", (checkpoint) => {
            if (hasArg("dry-run")) {
                return;
            }
            persistTrainCheckpoint(checkpoint);
        });

        await page.goto(`http://127.0.0.1:${port}/browser-runner/index.html`);
        await page.waitForFunction(
            () => typeof window.runBrowserTrain === "function",
        );

        const options = {
            bestValidationLoss: readBestValidationLoss(),
            maxRounds: readNumberArg("max-rounds"),
            batchSize: readNumberArg("batch-size"),
            validationInterval: readNumberArg("validation-interval"),
            validationSize: readNumberArg("validation-size"),
        };

        const result = await page.evaluate((options) => {
            return window.runBrowserTrain(options);
        }, options);

        if (!hasArg("dry-run")) {
            saveBestParams(result.bestParams);
            saveTrainingState(result.bestValidationLoss);
        }

        if (process.stdout.isTTY) {
            process.stdout.write("\n");
        }
        console.log(
            `Browser training finished. Best validation loss: ${result.bestValidationLoss.toFixed(4)}`,
        );
    } finally {
        await browser.close();
        server.close();
    }
}

await main();
