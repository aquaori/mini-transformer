const WORD_TOKEN_PATTERN = /[a-z]+(?:'[a-z]+)?|\d+|[.,!?;:'"()[\]\-/%&+]/g;
const NO_SPACE_BEFORE_PATTERN = /^[,.;:!?%)\]}]+$/;
const NO_SPACE_AFTER_PATTERN = /^[([{]$/;
const ENGLISH_ALLOWED_CHARS_PATTERN = /[^a-z0-9\s.,!?;:'"()[\]\-/%&+]/g;
const MULTI_SPACE_PATTERN = /\s+/g;

function stripDiacritics(text) {
    return text.normalize("NFKD").replace(/[\u0300-\u036f]/g, "");
}

function normalizeEnglishText(text) {
    return stripDiacritics(text)
        .replace(/[“”]/g, '"')
        .replace(/[‘’]/g, "'")
        .replace(/[‐‑–—−]/g, "-")
        .replace(/…/g, "...")
        .toLowerCase()
        .replace(ENGLISH_ALLOWED_CHARS_PATTERN, " ")
        .replace(MULTI_SPACE_PATTERN, " ")
        .trim();
}

function isMostlyEnglishText(text, minEnglishRatio = 0.7) {
    const compact = text.replace(/\s+/g, "");

    if (!compact) {
        return false;
    }

    const asciiLetters = (compact.match(/[A-Za-z]/g) ?? []).length;
    const allowedChars = (
        compact.match(/[A-Za-z0-9.,!?;:'"()[\]\-/%&+]/g) ?? []
    ).length;

    if (asciiLetters === 0) {
        return false;
    }

    return allowedChars / compact.length >= minEnglishRatio;
}

function splitWordTokens(text) {
    return normalizeEnglishText(text).match(WORD_TOKEN_PATTERN) ?? [];
}

function joinWordTokens(tokens) {
    let result = "";

    for (const token of tokens) {
        if (!result) {
            result = token;
            continue;
        }

        if (
            NO_SPACE_BEFORE_PATTERN.test(token) ||
            NO_SPACE_AFTER_PATTERN.test(result[result.length - 1])
        ) {
            result += token;
            continue;
        }

        result += ` ${token}`;
    }

    return result;
}

export { normalizeEnglishText, isMostlyEnglishText, splitWordTokens, joinWordTokens };
