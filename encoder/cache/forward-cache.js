const forwardCache = {
    data: {},

    reset() {
        this.data = {};
    },

    set(section, value) {
        this.data[section] = value;
    },

    get(section) {
        return this.data[section];
    },

    all() {
        return this.data;
    },
};

export { forwardCache };
