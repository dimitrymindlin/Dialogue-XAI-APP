-- Install the extension we just compiled

CREATE
EXTENSION IF NOT EXISTS vector;

CREATE TABLE items
(
    id        bigserial PRIMARY KEY,
    content   TEXT,
    embedding vector(1536)
);
