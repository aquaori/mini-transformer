function embeddingBackward(dEmbeddingOutput, tokenIds, vocabSize, dModel) {
    const dEmbeddingTable = Array.from({ length: vocabSize }, () =>
        new Array(dModel).fill(0),
    );

    for (let i = 0; i < tokenIds.length; i++) {
        const tokenId = tokenIds[i];

        for (let j = 0; j < dModel; j++) {
            dEmbeddingTable[tokenId][j] += dEmbeddingOutput[i][j];
        }
    }

    return {
        grads: {
            dEmbeddingTable,
        },
    };
}

export { embeddingBackward };
