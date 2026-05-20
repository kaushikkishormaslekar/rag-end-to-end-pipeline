package com.rag.gateway.chat;

import java.util.List;

import com.rag.gateway.rag.RagClient;
import com.rag.gateway.rag.RagQueryResponse;
import jakarta.validation.Valid;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

@RestController
@RequestMapping("/api/chat")
public class ChatController {
    private final RagClient ragGatewayClient;

    public ChatController(RagClient ragGatewayClient) {
        this.ragGatewayClient = ragGatewayClient;
    }

    @PostMapping
    public ChatResponse chat(@Valid @RequestBody ChatRequest request) {
        return toChatResponse(ragGatewayClient.query(request.message(), request.topK(), request.filters()));
    }

    @PostMapping(path = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter stream(@Valid @RequestBody ChatRequest request) {
        SseEmitter emitter = new SseEmitter(0L);

        Thread.ofVirtual().start(() -> {
            try {
                ragGatewayClient.streamQuery(
                        request.message(),
                        request.topK(),
                        request.filters(),
                        token -> send(emitter, "token", token)
                );
                send(emitter, "done", "");
                emitter.complete();
            } catch (Exception exception) {
                emitter.completeWithError(exception);
            }
        });

        return emitter;
    }

    private void send(SseEmitter emitter, String eventName, String data) {
        try {
            emitter.send(SseEmitter.event().name(eventName).data(data));
        } catch (Exception exception) {
            emitter.completeWithError(exception);
        }
    }

    private ChatResponse toChatResponse(RagQueryResponse response) {
        List<Citation> citations = response.citations() == null
                ? List.of()
                : response.citations()
                        .stream()
                        .map(citation -> new Citation(
                                citation.chunkId(),
                                citation.pageNumber(),
                                citation.source(),
                                citation.rerankScore()
                        ))
                        .toList();

        return new ChatResponse(
                response.answer(),
                response.abstained(),
                response.confidence(),
                citations
        );
    }
}
