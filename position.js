// 位置编码计算
// tokens: 输入的token序列
// dModel: 模型的维度
function getPositionalEmbeddings(tokens, dModel) {
    // 初始化位置编码数组
    const positionEmbeddings = [];

    // 遍历每个位置，计算对应的编码
    for (let pos = 0; pos < tokens.length; pos++) {
        // 初始化当前位置信息的子数组
        const subArray = [];

        // 计算每个维度的编码值
        for (let i = 0; i < dModel; i += 2) {
            // 计算分母，使用不同的频率进行编码
            const denominator = Math.pow(10000, i / dModel);

            // 计算sin和cos值，加上token原始值，并存储在子数组中
            subArray[i] = Math.sin(pos / denominator) + tokens[pos][i];

            // 如果当前维度的下一个位置在模型维度范围内，计算cos值
            if (i + 1 < dModel) {
                subArray[i + 1] =
                    Math.cos(pos / denominator) + tokens[pos][i + 1];
            }
        }

        positionEmbeddings.push(subArray);
    }

    return positionEmbeddings;
}

export { getPositionalEmbeddings };
