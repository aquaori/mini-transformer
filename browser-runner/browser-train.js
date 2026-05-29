import { trainModel } from "../encoder/scripts/training-core.js";
import { buildStatusText } from "../encoder/scripts/training-status.js";

const logElement = document.getElementById("log");

async function reportStatus(status) {
    logElement.textContent = buildStatusText(status);

    if (typeof window.reportTrainStatus === "function") {
        await window.reportTrainStatus(status);
    }
}

async function persistValidationCheckpoint(checkpoint) {
    if (typeof window.persistTrainCheckpoint === "function") {
        await window.persistTrainCheckpoint(checkpoint);
    }
}

async function trainInBrowser(options = {}) {
    return trainModel({
        maxRounds: options.maxRounds,
        batchSize: options.batchSize,
        validationInterval: options.validationInterval,
        validationSize: options.validationSize,
        earlyStopPatience: options.earlyStopPatience,
        bestValidationLoss: options.bestValidationLoss,
        reportStatus,
        onValidationComplete: persistValidationCheckpoint,
    });
}

window.runBrowserTrain = trainInBrowser;
