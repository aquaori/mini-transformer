import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";

const require = createRequire(import.meta.url);

function appendUniquePath(paths, value) {
    if (!value || paths.includes(value)) {
        return;
    }

    paths.push(value);
}

function getWindowsCudaBinCandidates() {
    const candidates = [];
    const envKeys = [
        "CUDA_PATH",
        "CUDA_HOME",
        "CUDA_ROOT",
        "CUDA_PATH_V12_8",
        "CUDA_PATH_V12_7",
        "CUDA_PATH_V12_6",
        "CUDA_PATH_V12_5",
        "CUDA_PATH_V12_4",
        "CUDA_PATH_V12_3",
        "CUDA_PATH_V12_2",
        "CUDA_PATH_V12_1",
        "CUDA_PATH_V12_0",
        "CUDA_PATH_V11_8",
    ];

    for (const key of envKeys) {
        const base = process.env[key];
        if (base) {
            appendUniquePath(candidates, path.join(base, "bin"));
        }
    }

    const defaultRoots = [
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
        "D:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
    ];

    for (const defaultRoot of defaultRoots) {
        if (!fs.existsSync(defaultRoot)) {
            continue;
        }

        const versions = fs
            .readdirSync(defaultRoot, { withFileTypes: true })
            .filter((entry) => entry.isDirectory())
            .map((entry) => entry.name)
            .sort((left, right) =>
                right.localeCompare(left, undefined, { numeric: true }),
            );

        for (const version of versions) {
            appendUniquePath(candidates, path.join(defaultRoot, version, "bin"));
        }
    }

    return candidates.filter((candidate) => fs.existsSync(candidate));
}

function ensureCudaDllSearchPath() {
    if (process.platform !== "win32") {
        return;
    }

    const delimiter = path.delimiter;
    const current = (process.env.PATH ?? "")
        .split(delimiter)
        .filter(Boolean);
    const additions = getWindowsCudaBinCandidates();
    let updated = false;

    for (const candidate of additions) {
        if (!current.includes(candidate)) {
            current.unshift(candidate);
            updated = true;
        }
    }

    if (updated) {
        process.env.PATH = current.join(delimiter);
    }
}

function getNativeCandidates() {
    const key = `${process.platform}-${process.arch}`;

    switch (key) {
        case "win32-x64":
            return [
                "@mni-ml/framework-win32-x64-msvc-cuda",
                "@mni-ml/framework-win32-x64-msvc-webgpu",
                "@mni-ml/framework-win32-x64-msvc",
            ];
        case "linux-x64":
            return [
                "@mni-ml/framework-linux-x64-gnu-webgpu",
                "@mni-ml/framework-linux-x64-gnu",
            ];
        case "darwin-arm64":
            return [
                "@mni-ml/framework-darwin-arm64-webgpu",
                "@mni-ml/framework-darwin-arm64",
            ];
        case "darwin-x64":
            return [
                "@mni-ml/framework-darwin-x64-webgpu",
                "@mni-ml/framework-darwin-x64",
            ];
        default:
            return ["@mni-ml/framework-win32-x64-msvc"];
    }
}

function getBackendName(packageName) {
    if (packageName.includes("cuda")) {
        return "cuda";
    }

    if (packageName.includes("webgpu")) {
        return "webgpu";
    }

    return "cpu";
}

function loadNative() {
    ensureCudaDllSearchPath();
    const candidates = getNativeCandidates();

    for (const packageName of candidates) {
        try {
            return {
                backend: getBackendName(packageName),
                native: require(packageName),
            };
        } catch {
            // Continue to the next candidate.
        }
    }

    throw new Error(
        `无法加载 @mni-ml/framework 原生后端，已尝试: ${candidates.join(", ")}`,
    );
}

const loaded = loadNative();
const native = loaded.native;
const backend = loaded.backend;

class Tensor {
    constructor(id, shape = null) {
        this._id = id;
        this._shape = shape ?? native.tensorShape(id).map(Number);
    }

    get shape() {
        return this._shape;
    }

    toFloat32() {
        return native.toFloat32(this._id);
    }

    free() {
        native.freeTensor(this._id);
    }

    matmul(other) {
        return new Tensor(native.matmul(this._id, other._id));
    }

    relu() {
        return new Tensor(native.relu(this._id));
    }

    static fromFloat32(data, shape) {
        return new Tensor(native.fromFloat32(data, shape.map(Number)), shape);
    }
}

function softmax(x, dim = -1) {
    return new Tensor(native.softmaxOp(x._id, dim));
}

function layerNorm(x, gamma, beta, eps = 1e-5) {
    return new Tensor(native.layernormOp(x._id, gamma._id, beta._id, eps));
}

export { Tensor, softmax, layerNorm, native, backend };
