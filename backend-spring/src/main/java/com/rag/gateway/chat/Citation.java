package com.rag.gateway.chat;

public record Citation(
        String chunkId,
        Object pageNumber,
        Object source,
        Double rerankScore
) {
}
