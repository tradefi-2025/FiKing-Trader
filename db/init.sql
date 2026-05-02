-- ─────────────────────────────────────────────────────────────────────────────
-- FiKing-Trader  PostgreSQL schema
-- Auto-executed by the postgres container on first boot (db/init.sql)
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS public.feature (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT,
    CONSTRAINT uq_feature_name UNIQUE (name)
);

CREATE TABLE IF NOT EXISTS public.agent (
    agent_id        SERIAL PRIMARY KEY,
    name            TEXT    NOT NULL,
    training_status TEXT    NOT NULL DEFAULT 'PENDING',
    version         TEXT,
    user_id         INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS public.agent_feature (
    agent_feature_id SERIAL PRIMARY KEY,
    agent_id         INTEGER NOT NULL REFERENCES public.agent (agent_id) ON DELETE CASCADE,
    feature_id       INTEGER NOT NULL REFERENCES public.feature (id)     ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS public.parameter_definition (
    parameter_definition_id SERIAL PRIMARY KEY,
    name          TEXT    NOT NULL,
    default_value TEXT,
    description   TEXT,
    min_value     TEXT,
    max_value     TEXT,
    type          TEXT    NOT NULL DEFAULT 'STRING',
    enum_values   TEXT,
    file_name     TEXT,
    required      BOOLEAN NOT NULL DEFAULT FALSE,
    feature_id    INTEGER NOT NULL REFERENCES public.feature (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS public.parameter_value (
    parameter_value_id      SERIAL PRIMARY KEY,
    value                   TEXT,
    agent_feature_id        INTEGER NOT NULL REFERENCES public.agent_feature (agent_feature_id) ON DELETE CASCADE,
    parameter_definition_id INTEGER NOT NULL REFERENCES public.parameter_definition (parameter_definition_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS public.signal (
    signal_id        SERIAL PRIMARY KEY,
    agent_id         INTEGER          NOT NULL REFERENCES public.agent (agent_id) ON DELETE CASCADE,
    signal_date      TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    estimated_action TEXT,
    signal           TEXT,
    probability      DOUBLE PRECISION,
    probabilities    JSONB            NOT NULL DEFAULT '{}',
    volume           DOUBLE PRECISION,
    notional         DOUBLE PRECISION,
    stop_loss_price  DOUBLE PRECISION,
    risk_amount      DOUBLE PRECISION,
    sizing_method    TEXT,
    warnings         TEXT[]           NOT NULL DEFAULT '{}',
    status           TEXT             NOT NULL DEFAULT 'NEW'
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_agent_user_id     ON public.agent (user_id);
CREATE INDEX IF NOT EXISTS idx_agent_status      ON public.agent (training_status);
CREATE INDEX IF NOT EXISTS idx_signal_agent_id   ON public.signal (agent_id);
CREATE INDEX IF NOT EXISTS idx_signal_date       ON public.signal (signal_date DESC);
CREATE INDEX IF NOT EXISTS idx_signal_status     ON public.signal (status);
CREATE INDEX IF NOT EXISTS idx_param_def_feature ON public.parameter_definition (feature_id);
CREATE INDEX IF NOT EXISTS idx_param_val_af      ON public.parameter_value (agent_feature_id);
