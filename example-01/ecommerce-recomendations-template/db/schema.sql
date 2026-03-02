CREATE TABLE IF NOT EXISTS products (
  id      SERIAL PRIMARY KEY,
  name     TEXT           NOT NULL,
  category TEXT           NOT NULL,
  price    NUMERIC(10, 2) NOT NULL,
  color    TEXT           NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
  id   INTEGER PRIMARY KEY,
  name TEXT    NOT NULL,
  age  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS purchases (
  user_id    INTEGER REFERENCES users(id)    ON DELETE CASCADE,
  product_id INTEGER REFERENCES products(id) ON DELETE CASCADE,
  PRIMARY KEY (user_id, product_id)
);

CREATE TABLE IF NOT EXISTS model_snapshots (
  id         INTEGER PRIMARY KEY DEFAULT 1,
  context    JSONB        NOT NULL,
  trained_at TIMESTAMPTZ  DEFAULT NOW()
);
