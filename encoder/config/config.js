export const TOKENIZER_TYPE = "bpe";
export const TOKENIZER_MIN_FREQUENCY = 2;
export const BPE_TARGET_VOCAB_SIZE = 2048;
export const BPE_MIN_PAIR_FREQUENCY = 2;

// 你当前 Encoder 的 dModel
// 如果你的 Wq/Wk/Wv 是 4×4，这里就必须是 4
export const D_MODEL = 256;
export const D_FF = D_MODEL * 2;
export const HEAD_NUMS = 2;
export const NUM_LAYERS = 2;

// embedding 初始化范围
export const EMBEDDING_SCALE = 0.02;
export const PARAM_INIT_SCALE = 0.02;

export const MAX_TRAIN_ROUND = 100;
export const EARLY_STOP_PATIENCE = 3;
export const LEARNING_RATE = 0.01;
export const VALIDATION_INTERVAL = 10;
export const BATCH_SIZE = 16;
