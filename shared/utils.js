import { init, matmul as accelMatMul } from "accel-gpu";

const GPU_MATMUL_MIN_OPS = 50_000;
const IS_BROWSER = typeof window !== "undefined";

let gpuContextPromise = null;
let frameworkPromise = null;
let loadedFramework = null;
let nodeComputeMode = "auto";

function getGpuContext() {
    if (!gpuContextPromise) {
        gpuContextPromise = init();
    }

    return gpuContextPromise;
}

async function getFramework() {
    if (IS_BROWSER) {
        return null;
    }

    if (!frameworkPromise) {
        frameworkPromise = import("./mni-framework-node.js");
    }

    return frameworkPromise;
}

async function detectComputeBackend() {
    if (IS_BROWSER) {
        try {
            const gpu = await getGpuContext();
            return gpu?.backendType ?? "cpu";
        } catch {
            return "cpu";
        }
    }

    try {
        const framework = loadedFramework ?? (await getFramework());
        if (framework) {
            loadedFramework = framework;
            return framework.backend ?? "cpu";
        }
    } catch {
        // Fall through to CPU fallback.
    }

    return "cpu";
}

function setNodeComputeMode(mode = "auto") {
    nodeComputeMode = mode;
}

function getNodeComputeMode() {
    return nodeComputeMode;
}

function shouldUseNodeAcceleratedOps() {
    return !IS_BROWSER && loadedFramework && nodeComputeMode !== "cpu";
}

function reshapeFlatArray(values, rows, cols) {
    return Array.from({ length: rows }, (_, row) =>
        Array.from(values.slice(row * cols, row * cols + cols)),
    );
}

function flattenMatrix(matrix) {
    return new Float32Array(matrix.flat());
}

function matMulCpu(A, B) {
    if (shouldUseNodeAcceleratedOps()) {
        return matMulSync(A, B);
    }

    return matMulFallback(A, B);
}

function matMulFallback(A, B) {
    const rowsA = A.length;
    const colsA = A[0].length;
    const colsB = B[0].length;
    const result = Array.from({ length: rowsA }, () =>
        new Array(colsB).fill(0),
    );

    for (let i = 0; i < rowsA; i++) {
        for (let k = 0; k < colsA; k++) {
            const a = A[i][k];

            for (let j = 0; j < colsB; j++) {
                result[i][j] += a * B[k][j];
            }
        }
    }

    return result;
}

async function matMul(A, B) {
    const rowsA = A.length;
    const colsA = A[0].length;
    const rowsB = B.length;
    const colsB = B[0].length;

    if (colsA !== rowsB) {
        throw new Error(
            `矩阵维度不匹配: ${rowsA}x${colsA} 不能乘 ${rowsB}x${colsB}`,
        );
    }

    const opCount = rowsA * colsA * colsB;

    if (!IS_BROWSER) {
        if (nodeComputeMode === "cpu") {
            return matMulFallback(A, B);
        }

        const framework = await getFramework();
        loadedFramework = framework;
        const { Tensor } = framework;
        const a = Tensor.fromFloat32(flattenMatrix(A), [rowsA, colsA]);
        const b = Tensor.fromFloat32(flattenMatrix(B), [rowsB, colsB]);
        const c = a.matmul(b);
        const result = reshapeFlatArray(c.toFloat32(), rowsA, colsB);

        a.free();
        b.free();
        c.free();

        return result;
    }

    if (opCount < GPU_MATMUL_MIN_OPS) {
        return matMulFallback(A, B);
    }

    const gpu = await getGpuContext();
    const a = gpu.array(flattenMatrix(A), [rowsA, colsA]);
    const b = gpu.array(flattenMatrix(B), [rowsB, colsB]);
    const c = await accelMatMul(gpu, a, b);
    const values = await c.toArray();
    const result = reshapeFlatArray(values, rowsA, colsB);

    a.dispose();
    b.dispose();
    c.dispose();

    return result;
}

function matMulSync(A, B) {
    const rowsA = A.length;
    const colsA = A[0].length;
    const rowsB = B.length;
    const colsB = B[0].length;

    if (colsA !== rowsB) {
        throw new Error(
            `矩阵维度不匹配: ${rowsA}x${colsA} 不能乘 ${rowsB}x${colsB}`,
        );
    }

    if (IS_BROWSER) {
        return matMulFallback(A, B);
    }

    if (!loadedFramework || nodeComputeMode === "cpu") {
        return matMulFallback(A, B);
    }

    const { Tensor } = loadedFramework;
    const a = Tensor.fromFloat32(flattenMatrix(A), [rowsA, colsA]);
    const b = Tensor.fromFloat32(flattenMatrix(B), [rowsB, colsB]);
    const c = a.matmul(b);
    const result = reshapeFlatArray(c.toFloat32(), rowsA, colsB);

    a.free();
    b.free();
    c.free();

    return result;
}

function addBias(matrix, bias) {
    return matrix.map((row) => row.map((value, index) => value + bias[index]));
}

function transpose(matrix) {
    return matrix[0].map((_, colIndex) => matrix.map((row) => row[colIndex]));
}

function addMatrices(A, B) {
    return A.map((row, i) => row.map((value, j) => value + B[i][j]));
}

function zerosLike(matrix) {
    return matrix.map((row) => new Array(row.length).fill(0));
}

function sumMatrices(matrices) {
    return matrices.reduce((acc, matrix) => addMatrices(acc, matrix));
}

function scaleMatrix(matrix, scale) {
    return matrix.map((row) => row.map((value) => value * scale));
}

function softmaxRowsCpu(matrix) {
    return matrix.map((row) => {
        const max = Math.max(...row);
        const exps = row.map((x) => Math.exp(x - max));
        const sum = exps.reduce((a, b) => a + b, 0);

        return exps.map((x) => x / sum);
    });
}

function softmaxRows(matrix) {
    if (IS_BROWSER) {
        return softmaxRowsCpu(matrix);
    }

    if (!loadedFramework || nodeComputeMode === "cpu") {
        return softmaxRowsCpu(matrix);
    }

    const { Tensor, softmax } = loadedFramework;
    const rows = matrix.length;
    const cols = matrix[0].length;
    const x = Tensor.fromFloat32(flattenMatrix(matrix), [rows, cols]);
    const y = softmax(x, -1);
    const result = reshapeFlatArray(y.toFloat32(), rows, cols);

    x.free();
    y.free();

    return result;
}

function relu(value) {
    return Math.max(0, value);
}

function reluMatrix(matrix) {
    if (IS_BROWSER) {
        return matrix.map((row) => row.map(relu));
    }

    if (!loadedFramework || nodeComputeMode === "cpu") {
        return matrix.map((row) => row.map(relu));
    }

    const { Tensor } = loadedFramework;
    const rows = matrix.length;
    const cols = matrix[0].length;
    const x = Tensor.fromFloat32(flattenMatrix(matrix), [rows, cols]);
    const y = x.relu();
    const result = reshapeFlatArray(y.toFloat32(), rows, cols);

    x.free();
    y.free();

    return result;
}

function layerNormRowCpu(x, gamma, beta, eps = 1e-6) {
    const mean = x.reduce((a, b) => a + b, 0) / x.length;
    const variance =
        x.reduce((sum, value) => sum + (value - mean) ** 2, 0) / x.length;
    const std = Math.sqrt(variance + eps);

    return x.map(
        (value, index) => ((value - mean) / std) * gamma[index] + beta[index],
    );
}

function layerNormRow(x, gamma, beta, eps = 1e-6) {
    if (IS_BROWSER) {
        return layerNormRowCpu(x, gamma, beta, eps);
    }

    if (!loadedFramework || nodeComputeMode === "cpu") {
        return layerNormRowCpu(x, gamma, beta, eps);
    }

    const { Tensor, layerNorm } = loadedFramework;
    const input = Tensor.fromFloat32(new Float32Array(x), [1, x.length]);
    const gammaTensor = Tensor.fromFloat32(new Float32Array(gamma), [x.length]);
    const betaTensor = Tensor.fromFloat32(new Float32Array(beta), [x.length]);
    const output = layerNorm(input, gammaTensor, betaTensor, eps);
    const result = Array.from(output.toFloat32());

    input.free();
    gammaTensor.free();
    betaTensor.free();
    output.free();

    return result;
}

function sumRows(matrix) {
    const cols = matrix[0].length;
    const result = new Array(cols).fill(0);

    for (const row of matrix) {
        for (let index = 0; index < cols; index++) {
            result[index] += row[index];
        }
    }

    return result;
}

getFramework()
    .then((framework) => {
        if (framework) {
            loadedFramework = framework;
        }
    })
    .catch(() => {
        // Node 侧框架不可用时回退到纯 JS。
    });

export {
    matMul,
    matMulSync,
    addBias,
    transpose,
    matMulCpu,
    addMatrices,
    zerosLike,
    sumMatrices,
    scaleMatrix,
    softmaxRows,
    reluMatrix,
    layerNormRow,
    sumRows,
    detectComputeBackend,
    setNodeComputeMode,
    getNodeComputeMode,
};
