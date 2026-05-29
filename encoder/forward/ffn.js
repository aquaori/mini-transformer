// ffn 前馈神经网络
import { matMul, addBias, reluMatrix } from "../../shared/utils.js";

// 前馈网络主体
async function ffn(x, W1, b1, W2, b2) {
    const hidden = addBias(await matMul(x, W1), b1);

    const activated = reluMatrix(hidden);

    const output = addBias(await matMul(activated, W2), b2);
    const cache = {
        input: x,
        W1,
        b1,
        W2,
        b2,
        hidden,
        activated,
        output,
    };

    return { output, cache };
}

export { ffn };
