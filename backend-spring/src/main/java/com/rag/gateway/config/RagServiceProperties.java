package com.rag.gateway.config;

import java.time.Duration;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "rag.service")
public record RagServiceProperties(
        String baseUrl,
        Duration requestTimeout
) {
}
