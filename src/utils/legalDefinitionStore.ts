// src/utils/legalDefinitionStore.ts

import { create } from "zustand";

interface LegalDefinitionState {
    term: string;
    definition: string;
    setDefinition: (term: string, definition: string) => void;
    clear: () => void;
}

export const useLegalDefinitionStore = create<LegalDefinitionState>((set) => ({
    term: "",
    definition: "",
    setDefinition: (term, definition) => set({ term, definition }),
    clear: () => set({ term: "", definition: "" }),
}));
