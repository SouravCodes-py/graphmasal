-- ============================================================
-- GraphMASAL — Semantic Memory Store
-- Run this once in the Supabase SQL Editor to set up
-- the table, index, and RPC function for memory_agent.py.
-- ============================================================

-- 1. Enable the pgvector extension (idempotent)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create the semantic_memories table
CREATE TABLE IF NOT EXISTS semantic_memories (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    student_id  TEXT        NOT NULL,
    summary     TEXT        NOT NULL,
    embedding   VECTOR(3072) NOT NULL,   -- gemini-embedding-001 dimensions
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 3. Add an ivfflat index for fast cosine-similarity search.
--    lists = 100 is a good starting point; tune after the table
--    grows past ~10 k rows. The index must be built AFTER some
--    rows exist, but CREATE INDEX IF NOT EXISTS keeps it safe to
--    re-run later.
CREATE INDEX IF NOT EXISTS idx_semantic_memories_embedding
    ON semantic_memories
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- 4. Index on student_id for fast filtering
CREATE INDEX IF NOT EXISTS idx_semantic_memories_student_id
    ON semantic_memories (student_id);

-- 5. RPC function: match_semantic_memories
--    Called by memory_agent.fetch() via supabase.rpc().
--    Returns the top-k most similar summaries for a given student.
CREATE OR REPLACE FUNCTION match_semantic_memories(
    p_student_id  TEXT,
    p_embedding   VECTOR(3072),
    p_match_count INT DEFAULT 3
)
RETURNS TABLE (
    id          BIGINT,
    student_id  TEXT,
    summary     TEXT,
    similarity  FLOAT
)
LANGUAGE sql STABLE
AS $$
    SELECT
        sm.id,
        sm.student_id,
        sm.summary,
        1 - (sm.embedding <=> p_embedding) AS similarity
    FROM semantic_memories sm
    WHERE sm.student_id = p_student_id
    ORDER BY sm.embedding <=> p_embedding   -- ascending distance = descending similarity
    LIMIT p_match_count;
$$;
