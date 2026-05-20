package com.rag.gateway.rag;

import com.fasterxml.jackson.annotation.JsonProperty;

public record RagCitation(
        @JsonProperty("chunk_id") String chunkId,
        @JsonProperty("page_number") Object pageNumber,
        Object source,
        @JsonProperty("rerank_score") Double rerankScore
) {
}
