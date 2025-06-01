// Shared state between functions
let currentVectorStore = null;

export const getVectorStore = () => currentVectorStore;
export const setVectorStore = (store) => {
    currentVectorStore = store;
}; 