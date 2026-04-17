-- Wire audit tracking + updated_at triggers for xau_gold config tables.
--
-- Before this migration: xau_gold_global_config and xau_gold_pair_overrides had
-- no triggers, so updates silently changed values with no audit row written and
-- updated_at was only bumped when callers happened to set it manually.
--
-- After: every UPDATE fires a BEFORE trigger that bumps updated_at, and an
-- AFTER trigger that writes one row per changed tracked field into
-- xau_gold_config_audit.

-- ---------------------------------------------------------------------------
-- global_config: each row represents one parameter (key/value store).
-- Log when parameter_value or is_active changes.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION xau_gold_global_config_audit_trigger()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.parameter_value IS DISTINCT FROM OLD.parameter_value THEN
        INSERT INTO xau_gold_config_audit (
            table_name, record_id, parameter_name,
            old_value, new_value, change_type,
            changed_by, change_reason, config_set
        ) VALUES (
            'xau_gold_global_config', NEW.id, NEW.parameter_name,
            OLD.parameter_value, NEW.parameter_value, 'UPDATE',
            COALESCE(NEW.updated_by, 'system'), NEW.change_reason, NEW.config_set
        );
    END IF;

    IF NEW.is_active IS DISTINCT FROM OLD.is_active THEN
        INSERT INTO xau_gold_config_audit (
            table_name, record_id, parameter_name,
            old_value, new_value, change_type,
            changed_by, change_reason, config_set
        ) VALUES (
            'xau_gold_global_config', NEW.id, 'is_active',
            OLD.is_active::text, NEW.is_active::text, 'UPDATE',
            COALESCE(NEW.updated_by, 'system'), NEW.change_reason, NEW.config_set
        );
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ---------------------------------------------------------------------------
-- pair_overrides: wide table — diff row-as-jsonb and emit one audit row per
-- changed tracked column.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION xau_gold_pair_overrides_audit_trigger()
RETURNS TRIGGER AS $$
DECLARE
    old_j JSONB := to_jsonb(OLD);
    new_j JSONB := to_jsonb(NEW);
    k TEXT;
    tracked_cols CONSTANT TEXT[] := ARRAY[
        'pair_name', 'pip_size',
        'fixed_stop_loss_pips', 'fixed_take_profit_pips',
        'min_confidence', 'max_confidence',
        'sl_atr_multiplier', 'rr_ratio',
        'signal_cooldown_minutes', 'parameter_overrides',
        'is_enabled', 'is_traded', 'monitor_only', 'notes'
    ];
BEGIN
    FOREACH k IN ARRAY tracked_cols LOOP
        IF old_j->k IS DISTINCT FROM new_j->k THEN
            INSERT INTO xau_gold_config_audit (
                table_name, record_id, parameter_name,
                old_value, new_value, change_type,
                changed_by, change_reason, config_set
            ) VALUES (
                'xau_gold_pair_overrides', NEW.id, k,
                old_j->>k, new_j->>k, 'UPDATE',
                COALESCE(NEW.updated_by, 'system'), NEW.change_reason, NEW.config_set
            );
        END IF;
    END LOOP;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ---------------------------------------------------------------------------
-- Attach triggers (idempotent via DROP IF EXISTS).
-- ---------------------------------------------------------------------------

-- BEFORE UPDATE: auto-bump updated_at. Uses the shared update_updated_at_column()
-- function already defined by the smc_simple migration.
DROP TRIGGER IF EXISTS update_xau_gold_global_config_updated_at ON xau_gold_global_config;
CREATE TRIGGER update_xau_gold_global_config_updated_at
    BEFORE UPDATE ON xau_gold_global_config
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_xau_gold_pair_overrides_updated_at ON xau_gold_pair_overrides;
CREATE TRIGGER update_xau_gold_pair_overrides_updated_at
    BEFORE UPDATE ON xau_gold_pair_overrides
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- AFTER UPDATE: write audit rows.
DROP TRIGGER IF EXISTS xau_gold_global_config_audit ON xau_gold_global_config;
CREATE TRIGGER xau_gold_global_config_audit
    AFTER UPDATE ON xau_gold_global_config
    FOR EACH ROW
    EXECUTE FUNCTION xau_gold_global_config_audit_trigger();

DROP TRIGGER IF EXISTS xau_gold_pair_overrides_audit ON xau_gold_pair_overrides;
CREATE TRIGGER xau_gold_pair_overrides_audit
    AFTER UPDATE ON xau_gold_pair_overrides
    FOR EACH ROW
    EXECUTE FUNCTION xau_gold_pair_overrides_audit_trigger();
