-- Run this once against the postgres container to set up the logs table.
-- docker exec -i <postgres-container> psql -U postgres -d logs < schema.sql
-- (or just paste into psql / a GUI client connected to localhost:5432)

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS logs (
    id          SERIAL PRIMARY KEY,
    log_line    TEXT NOT NULL,
    ts          TIMESTAMP NOT NULL,
    host        TEXT,
    source_ip   TEXT,
    user_id     TEXT,
    embedding   VECTOR(768)  -- matches Gemini's text-embedding-004 output dim
);

-- Speeds up the correlate_node queries (entity + time-window lookups)
CREATE INDEX IF NOT EXISTS idx_logs_source_ip ON logs (source_ip);
CREATE INDEX IF NOT EXISTS idx_logs_user_id ON logs (user_id);
CREATE INDEX IF NOT EXISTS idx_logs_host ON logs (host);
CREATE INDEX IF NOT EXISTS idx_logs_ts ON logs (ts);
-- Approximate nearest-neighbour index for search_logs_node (cosine distance).
-- HNSW needs no training step, so it stays valid after the table is reloaded.
CREATE INDEX IF NOT EXISTS idx_logs_embedding ON logs
    USING hnsw (embedding vector_cosine_ops);
