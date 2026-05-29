function formatDuration(ms) {
    if (!Number.isFinite(ms) || ms < 0) {
        return "--:--";
    }

    const totalSeconds = Math.floor(ms / 1000);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;

    if (hours > 0) {
        return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
    }

    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function renderProgressBar(progress, width = 28) {
    const clamped = Math.max(0, Math.min(1, progress ?? 0));
    const filled = Math.round(clamped * width);
    return `${"=".repeat(filled)}${"-".repeat(width - filled)}`;
}

function buildStatusLines(status) {
    const percent = `${((status.progress ?? 0) * 100).toFixed(1)}%`;
    const validationInfo =
        status.state === "validating"
            ? ` | Validate ${status.validationIndex}/${status.validationTotal}`
            : "";

    return [
        "Training Status",
        `[${renderProgressBar(status.progress)}] ${percent} | Round ${status.round}/${status.maxRounds}${validationInfo}`,
        `State: ${status.state} | Backend: ${status.backend ?? "--"}`,
        `Elapsed: ${formatDuration(status.elapsedMs)} | ETA: ${formatDuration(status.etaMs)} | Batch: ${status.batchSize ?? "--"}`,
        `Train Loss: ${status.trainLossText ?? "--"} | Validation Loss: ${status.validationLossText ?? "--"} | Best Loss: ${status.bestLossText ?? "--"}`,
        `Patience: ${status.badSteps ?? 0}/${status.patience ?? 0}`,
        status.profileLine ? `Profile: ${status.profileLine}` : "Profile: --",
    ];
}

function buildStatusText(status) {
    return buildStatusLines(status).join("\n");
}

function createConsoleStatusRenderer(output = process.stdout) {
    let lastPanelLines = 0;

    return function renderConsoleStatus(status) {
        const lines = buildStatusLines(status);

        if (output.isTTY) {
            if (lastPanelLines > 0) {
                output.write(`\x1b[${lastPanelLines}F`);
            }
            output.write("\x1b[0J");
            output.write(`${lines.join("\n")}\n`);
        } else {
            console.log(lines.join(" | "));
        }

        lastPanelLines = lines.length;
    };
}

export {
    formatDuration,
    renderProgressBar,
    buildStatusLines,
    buildStatusText,
    createConsoleStatusRenderer,
};
