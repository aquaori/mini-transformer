// 0~255 表示 UTF-8 byte token
export const BYTE_VOCAB_SIZE = 256;

// 特殊 token
export const BOS_TOKEN_ID = 256;
export const EOS_TOKEN_ID = 257;

// 总词表大小：0~255 + <bos> + <eos>
export const VOCAB_SIZE = 258;

// 你当前 Encoder 的 dModel
// 如果你的 Wq/Wk/Wv 是 4×4，这里就必须是 4
export const D_MODEL = 4;

// embedding 初始化范围
export const EMBEDDING_SCALE = 0.02;
