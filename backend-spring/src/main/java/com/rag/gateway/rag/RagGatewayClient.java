package com.rag.gateway.rag;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.function.Consumer;

import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientResponseException;

@Service
public class RagGatewayClient implements RagClient {
    private final RestClient ragRestClient;

    public RagGatewayClient(RestClient ragRestClient) {
        this.ragRestClient = ragRestClient;
    }

    @Override
    public RagQueryResponse query(String message, Integer topK, Map<String, Object> filters) {
        RagQueryRequest request = new RagQueryRequest(message, topK, filters, true);
        try {
            return ragRestClient
                    .post()
                    .uri("/query")
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(request)
                    .retrieve()
                    .body(RagQueryResponse.class);
        } catch (RestClientResponseException exception) {
            throw new RagServiceException(exception.getStatusCode(), exception.getResponseBodyAsString());
        }
    }

    @Override
    public void streamQuery(
            String message,
            Integer topK,
            Map<String, Object> filters,
            Consumer<String> eventConsumer
    ) {
        RagQueryRequest request = new RagQueryRequest(message, topK, filters, true);
        try {
            ragRestClient
                .post()
                .uri("/query/stream")
                .accept(MediaType.TEXT_EVENT_STREAM)
                .contentType(MediaType.APPLICATION_JSON)
                .body(request)
                .exchange((ignoredRequest, response) -> {
                    if (response.getStatusCode().isError()) {
                        throw new RagServiceException(response.getStatusCode(), "RAG service stream failed");
                    }

                    try (BufferedReader reader = new BufferedReader(
                            new InputStreamReader(response.getBody(), StandardCharsets.UTF_8))) {
                        String line;
                        while ((line = reader.readLine()) != null) {
                            if (line.startsWith("data:")) {
                                eventConsumer.accept(line.substring("data:".length()).trim());
                            }
                        }
                    }
                    return null;
                });
        } catch (RestClientResponseException exception) {
            throw new RagServiceException(exception.getStatusCode(), exception.getResponseBodyAsString());
        }
    }
}
