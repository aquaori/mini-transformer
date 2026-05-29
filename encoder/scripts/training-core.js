import {
    BATCH_SIZE,
    EARLY_STOP_PATIENCE,
    MAX_TRAIN_ROUND,
    VALIDATION_INTERVAL,
} from "../config/config.js";
import { trainSets, validationSets } from "../data/dataSets.js";
import { forwardCache } from "../cache/forward-cache.js";
import { Encoder } from "../index.js";
import { backward, crossEntropyLoss } from "../backward/backward.js";
import {
    applyAllGradients,
    averageBackwardStates,
    cloneModelParams,
} from "../backward/paramsUpdate.js";
import {
    Wq,
    Wk,
    Wv,
    W1,
    b1,
    W2,
    b2,
    Wo,
    gamma,
    beta,
    headNums,
} from "../params/data.js";
import { embeddingMatrix } from "../params/embedding-table.js";
import { W_vocab, b_vocab } from "../params/vocab_head.js";
import {
    detectComputeBackend,
    getNodeComputeMode,
    setNodeComputeMode,
} from "../../shared/utils.js";

function createInitialParams() {
    return cloneModelParams({
        Wq,
        Wk,
        Wv,
        W1,
        b1,
        W2,
        b2,
        Wo,
        gamma,
        beta,
        headNums,
        W_vocab,
        b_vocab,
        embeddingMatrix,
    });
}

async function runEncoder(inputText, params) {
    return Encoder(
        inputText,
        params.Wq,
        params.Wk,
        params.Wv,
        params.W1,
        params.b1,
        params.W2,
        params.b2,
        params.Wo,
        params.gamma,
        params.beta,
        params.headNums,
        params.W_vocab,
        params.b_vocab,
    );
}

function buildEta(startedAt, progress) {
    const elapsedMs = Date.now() - startedAt;
    const etaMs =
        progress > 0 ? (elapsedMs / progress) * (1 - progress) : Infinity;

    return { elapsedMs, etaMs };
}

function createProfilingState() {
    return {
        trainForwardMs: 0,
        trainBackwardMs: 0,
        trainAverageMs: 0,
        trainUpdateMs: 0,
        validationForwardMs: 0,
        validationLossMs: 0,
        trainSamples: 0,
        validationSamples: 0,
    };
}

function buildProfileLine(profile) {
    const segments = [
        `fwd ${profile.trainForwardMs.toFixed(0)}ms`,
        `bwd ${profile.trainBackwardMs.toFixed(0)}ms`,
        `merge ${profile.trainAverageMs.toFixed(0)}ms`,
        `update ${profile.trainUpdateMs.toFixed(0)}ms`,
        `val-fwd ${profile.validationForwardMs.toFixed(0)}ms`,
    ];

    return segments.join(" | ");
}

async function evaluateValidation(
    params,
    validationSize,
    statusContext,
    reportStatus,
    profile,
) {
    const losses = [];
    const texts = validationSets.slice(0, validationSize);
    const previousMode = getNodeComputeMode();
    setNodeComputeMode("cpu");

    try {
        for (let index = 0; index < texts.length; index++) {
            const validationForwardStartedAt = Date.now();
            const { probabilities } = await runEncoder(texts[index], params);
            profile.validationForwardMs +=
                Date.now() - validationForwardStartedAt;

            const validationLossStartedAt = Date.now();
            const { originalTokenIds } = forwardCache.get("input");
            losses.push(crossEntropyLoss(probabilities, originalTokenIds));
            profile.validationLossMs += Date.now() - validationLossStartedAt;
            profile.validationSamples++;

            const avgLoss = losses.reduce((a, b) => a + b, 0) / losses.length;
            const progress =
                ((statusContext.round - 1) +
                    (index + 1) / Math.max(1, texts.length)) /
                Math.max(1, statusContext.maxRounds);
            const { elapsedMs, etaMs } = buildEta(
                statusContext.startedAt,
                progress,
            );

            await reportStatus({
                ...statusContext,
                backend: "cpu",
                state: "validating",
                progress,
                elapsedMs,
                etaMs,
                validationIndex: index + 1,
                validationTotal: texts.length,
                validationLoss: avgLoss,
                validationLossText: avgLoss.toFixed(4),
                profileLine: buildProfileLine(profile),
            });
        }
    } finally {
        setNodeComputeMode(previousMode);
    }

    return losses.reduce((a, b) => a + b, 0) / losses.length;
}

async function trainModel(options = {}) {
    const maxRounds = options.maxRounds ?? MAX_TRAIN_ROUND;
    const validationInterval =
        options.validationInterval ?? VALIDATION_INTERVAL;
    const validationSize = options.validationSize ?? validationSets.length;
    const earlyStopPatience = options.earlyStopPatience ?? EARLY_STOP_PATIENCE;
    const batchSize = options.batchSize ?? BATCH_SIZE;
    const reportStatus = options.reportStatus ?? (async () => {});
    const onBestParams = options.onBestParams ?? (async () => {});
    const onValidationComplete =
        options.onValidationComplete ?? (async () => {});
    let bestValidationLoss = Number.isFinite(options.bestValidationLoss)
        ? options.bestValidationLoss
        : Infinity;
    let currentParams = cloneModelParams(
        options.initialParams ?? createInitialParams(),
    );
    let bestParams = cloneModelParams(currentParams);
    let badSteps = 0;
    let finalRound = 0;
    let stoppedEarly = false;
    let lastTrainLoss = null;
    let lastValidationLoss = null;
    const startedAt = Date.now();
    const backend = await detectComputeBackend();
    const profile = createProfilingState();

    for (let round = 1; round <= maxRounds; round++) {
        finalRound = round;
        const batchBackwardStates = [];
        const batchLosses = [];

        for (let batchIndex = 0; batchIndex < batchSize; batchIndex++) {
            const inputText =
                trainSets[Math.floor(Math.random() * trainSets.length)];
            const forwardStartedAt = Date.now();
            const { probabilities } = await runEncoder(inputText, currentParams);
            profile.trainForwardMs += Date.now() - forwardStartedAt;
            const backwardStartedAt = Date.now();
            const backwardResult = backward(probabilities, currentParams.W_vocab);
            profile.trainBackwardMs += Date.now() - backwardStartedAt;

            batchBackwardStates.push(backwardResult.grads);
            batchLosses.push(backwardResult.loss);
            profile.trainSamples++;
        }

        const averageStartedAt = Date.now();
        const averagedBackwardState = averageBackwardStates(batchBackwardStates);
        profile.trainAverageMs += Date.now() - averageStartedAt;
        const updateStartedAt = Date.now();
        currentParams = applyAllGradients(
            currentParams,
            averagedBackwardState,
        );
        profile.trainUpdateMs += Date.now() - updateStartedAt;

        const trainLoss =
            batchLosses.reduce((sum, loss) => sum + loss, 0) /
            batchLosses.length;
        lastTrainLoss = trainLoss;

        const progress = round / maxRounds;
        const { elapsedMs, etaMs } = buildEta(startedAt, progress);

        await reportStatus({
            backend,
            state: "training",
            round,
            maxRounds,
            progress,
            elapsedMs,
            etaMs,
            trainLoss,
            trainLossText: trainLoss.toFixed(4),
            validationLoss: lastValidationLoss,
            validationLossText: Number.isFinite(lastValidationLoss)
                ? lastValidationLoss.toFixed(4)
                : "--",
            bestLoss: bestValidationLoss,
            bestLossText: Number.isFinite(bestValidationLoss)
                ? bestValidationLoss.toFixed(4)
                : "--",
            batchSize,
            badSteps,
            patience: earlyStopPatience,
            validationIndex: 0,
            validationTotal: validationSize,
            profileLine: buildProfileLine(profile),
        });

        if (round % validationInterval === 0) {
            const avgValidationLoss = await evaluateValidation(
                currentParams,
                validationSize,
                {
                    backend,
                    round,
                    maxRounds,
                    startedAt,
                    trainLoss,
                    trainLossText: trainLoss.toFixed(4),
                    bestLoss: bestValidationLoss,
                    bestLossText: Number.isFinite(bestValidationLoss)
                        ? bestValidationLoss.toFixed(4)
                        : "--",
                    batchSize,
                    badSteps,
                    patience: earlyStopPatience,
                },
                reportStatus,
                profile,
            );

            if (avgValidationLoss < bestValidationLoss) {
                bestValidationLoss = avgValidationLoss;
                badSteps = 0;
                bestParams = cloneModelParams(currentParams);
                await onBestParams(bestParams, bestValidationLoss);
            } else {
                badSteps++;
            }

            lastValidationLoss = avgValidationLoss;
            const postValidationProgress = round / maxRounds;
            const validationTiming = buildEta(startedAt, postValidationProgress);

            await reportStatus({
                backend,
                state: "validation-complete",
                round,
                maxRounds,
                progress: postValidationProgress,
                elapsedMs: validationTiming.elapsedMs,
                etaMs: validationTiming.etaMs,
                trainLoss,
                trainLossText: trainLoss.toFixed(4),
                validationLoss: avgValidationLoss,
                validationLossText: avgValidationLoss.toFixed(4),
                bestLoss: bestValidationLoss,
                bestLossText: bestValidationLoss.toFixed(4),
                batchSize,
                badSteps,
                patience: earlyStopPatience,
                validationIndex: validationSize,
                validationTotal: validationSize,
                profileLine: buildProfileLine(profile),
            });

            await onValidationComplete({
                currentParams: cloneModelParams(currentParams),
                bestParams: cloneModelParams(bestParams),
                bestValidationLoss,
                validationLoss: avgValidationLoss,
                round,
            });

            if (badSteps >= earlyStopPatience) {
                stoppedEarly = true;
                await reportStatus({
                    backend,
                    state: "stopped",
                    round,
                    maxRounds,
                    progress: postValidationProgress,
                    elapsedMs: validationTiming.elapsedMs,
                    etaMs: 0,
                    trainLoss,
                    trainLossText: trainLoss.toFixed(4),
                    validationLoss: avgValidationLoss,
                    validationLossText: avgValidationLoss.toFixed(4),
                    bestLoss: bestValidationLoss,
                    bestLossText: bestValidationLoss.toFixed(4),
                    batchSize,
                    badSteps,
                    patience: earlyStopPatience,
                    validationIndex: validationSize,
                    validationTotal: validationSize,
                    profileLine: buildProfileLine(profile),
                });
                break;
            }
        }
    }

    if (!stoppedEarly) {
        await reportStatus({
            backend,
            state: "completed",
            round: finalRound,
            maxRounds,
            progress: finalRound / maxRounds,
            elapsedMs: Date.now() - startedAt,
            etaMs: 0,
            trainLoss: lastTrainLoss,
            trainLossText: Number.isFinite(lastTrainLoss)
                ? lastTrainLoss.toFixed(4)
                : "--",
            validationLoss: lastValidationLoss,
            validationLossText: Number.isFinite(lastValidationLoss)
                ? lastValidationLoss.toFixed(4)
                : "--",
            bestLoss: bestValidationLoss,
            bestLossText: Number.isFinite(bestValidationLoss)
                ? bestValidationLoss.toFixed(4)
                : "--",
            batchSize,
            badSteps,
            patience: earlyStopPatience,
            validationIndex: validationSize,
            validationTotal: validationSize,
            profileLine: buildProfileLine(profile),
        });
    }

    return {
        bestParams,
        bestValidationLoss,
        currentParams,
        stoppedEarly,
        finalRound,
        lastTrainLoss,
        lastValidationLoss,
        profile,
    };
}

export { createInitialParams, runEncoder, trainModel };
