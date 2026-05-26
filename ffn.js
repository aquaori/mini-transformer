// ffn 前馈神经网络
import { matMul, addBias } from "./utils.js";

// ReLu 激活函数
function relu(x) {
    return Math.max(0, x);
}

// ReLU 矩阵乘法
function reluMatrix(matrix) {
    return matrix.map((row) => row.map(relu));
}

// 前馈网络主体
function ffn(x, W1, b1, W2, b2) {
    const hidden = addBias(matMul(x, W1), b1);

    const activated = reluMatrix(hidden);

    const output = addBias(matMul(activated, W2), b2);

    return output;
}

export { ffn };
