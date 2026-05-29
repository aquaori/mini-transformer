import { layerNormRow } from "../../shared/utils.js";

function add(x, subLayerOutput) {
    return x.map((value, index) => value + subLayerOutput[index]);
}

function norm(x, gamma, beta, eps = 1e-6) {
    return layerNormRow(x, gamma, beta, eps);
}

export { add, norm };
