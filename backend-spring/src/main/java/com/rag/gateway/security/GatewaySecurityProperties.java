package com.rag.gateway.security;

import java.time.Duration;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "gateway.security")
public record GatewaySecurityProperties(
        String issuer,
        String jwtSecret,
        Duration tokenTtl,
        String demoUsername,
        String demoPassword
) {
    public GatewaySecurityProperties {
        issuer = defaultString(issuer, "rag-spring-gateway");
        jwtSecret = defaultString(jwtSecret, "dev-only-change-me-32-byte-secret-key");
        tokenTtl = tokenTtl == null ? Duration.ofHours(1) : tokenTtl;
        demoUsername = defaultString(demoUsername, "demo");
        demoPassword = defaultString(demoPassword, "demo123");

        if (jwtSecret.getBytes().length < 32) {
            throw new IllegalArgumentException("gateway.security.jwt-secret must be at least 32 bytes for HS256");
        }
    }

    private static String defaultString(String value, String fallback) {
        return value == null || value.isBlank() ? fallback : value;
    }
}
