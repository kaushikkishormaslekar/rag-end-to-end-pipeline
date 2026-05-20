export type Citation = {
  chunkId: string;
  pageNumber: number | null;
  source: string | null;
  rerankScore: number | null;
};

export type ChatResponse = {
  answer: string;
  abstained: boolean;
  confidence: number | null;
  citations: Citation[];
};

export type ChatRequest = {
  message: string;
  topK: number;
  filters: Record<string, unknown>;
};

export type DocumentRecord = {
  document_id: string;
  filename: string;
  content_type: string | null;
  size_bytes: number;
  source: string;
  status: string;
  chunk_count: number;
  created_at: string;
};

export type DocumentListResponse = {
  documents: DocumentRecord[];
};
