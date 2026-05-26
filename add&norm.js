// 残差连接和归一化处理
import { gamma, beta } from "./data.js";

function add(x, subLayerOutput) {
    return x.map((value, index) => value + subLayerOutput[index]);
}

function norm(x) {
    const mean = x.reduce((sum, value) => sum + value, 0) / x.length;
    const variance =
        x.reduce((sum, value) => sum + (value - mean) ** 2, 0) / x.length;
    return x.map(
        (value, index) =>
            (gamma[index] * (value - mean)) / Math.sqrt(variance + 1e-6) +
            beta[index],
    );
}

export { add, norm };
