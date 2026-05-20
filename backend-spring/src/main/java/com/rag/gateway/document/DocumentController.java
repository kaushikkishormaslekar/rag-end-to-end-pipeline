package com.rag.gateway.document;

import java.io.IOException;

import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.ContentDisposition;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/documents")
public class DocumentController {
    private final RestClient ragRestClient;

    public DocumentController(RestClient ragRestClient) {
        this.ragRestClient = ragRestClient;
    }

    @GetMapping
    public ResponseEntity<String> listDocuments() {
        return forward(() -> ragRestClient
                .get()
                .uri("/documents")
                .retrieve()
                .toEntity(String.class));
    }

    @GetMapping("/{documentId}")
    public ResponseEntity<String> getDocument(@PathVariable String documentId) {
        return forward(() -> ragRestClient
                .get()
                .uri("/documents/{documentId}", documentId)
                .retrieve()
                .toEntity(String.class));
    }

    @PostMapping(path = "/upload", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<String> uploadDocument(
            @RequestParam("file") MultipartFile file,
            @RequestParam(value = "source", required = false) String source
    ) throws IOException {
        String filename = file.getOriginalFilename() == null ? "document" : file.getOriginalFilename();
        HttpHeaders fileHeaders = new HttpHeaders();
        fileHeaders.setContentDisposition(ContentDisposition.formData()
                .name("file")
                .filename(filename)
                .build());
        fileHeaders.setContentType(file.getContentType() == null
                ? MediaType.APPLICATION_OCTET_STREAM
                : MediaType.parseMediaType(file.getContentType()));

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", new HttpEntity<>(
                new NamedByteArrayResource(file.getBytes(), filename),
                fileHeaders));
        if (source != null && !source.isBlank()) {
            body.add("source", source);
        }

        return forward(() -> ragRestClient
                .post()
                .uri("/documents/upload")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(body)
                .retrieve()
                .toEntity(String.class));
    }

    @DeleteMapping("/{documentId}")
    public ResponseEntity<String> deleteDocument(@PathVariable String documentId) {
        return forward(() -> ragRestClient
                .delete()
                .uri("/documents/{documentId}", documentId)
                .retrieve()
                .toEntity(String.class));
    }

    private ResponseEntity<String> forward(RestCall call) {
        try {
            ResponseEntity<String> response = call.execute();
            return ResponseEntity
                    .status(response.getStatusCode())
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(response.getBody());
        } catch (RestClientResponseException exception) {
            return ResponseEntity
                    .status(HttpStatusCode.valueOf(exception.getStatusCode().value()))
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(exception.getResponseBodyAsString());
        }
    }

    @FunctionalInterface
    private interface RestCall {
        ResponseEntity<String> execute();
    }

    private static class NamedByteArrayResource extends ByteArrayResource {
        private final String filename;

        NamedByteArrayResource(byte[] byteArray, String filename) {
            super(byteArray);
            this.filename = filename == null || filename.isBlank() ? "document" : filename;
        }

        @Override
        public String getFilename() {
            return filename;
        }
    }
}
