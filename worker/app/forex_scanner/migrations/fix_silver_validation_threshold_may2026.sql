-- Fix validate_price_data: raise Silver effective_threshold from 2.0 to 50.0.
-- The old threshold caused false SUSPICIOUS_MOVE warnings on every normal
-- 1-minute Silver candle (~$6-10 moves at current prices ~$7500/contract).
-- Silver now matches Gold's threshold (50.0), with suspicious fires at 3x = $150.
-- Applies to: forex database.

CREATE OR REPLACE FUNCTION validate_price_data(
    p_epic VARCHAR,
    p_timeframe INTEGER,
    p_candle_time TIMESTAMP,
    p_close NUMERIC,
    p_data_source VARCHAR,
    p_threshold_pips NUMERIC DEFAULT 10.0
) RETURNS TABLE(is_valid BOOLEAN, quality_score NUMERIC, validation_flags TEXT[], warning_message TEXT) AS $$
DECLARE
    recent_close DECIMAL;
    price_diff_pips DECIMAL;
    effective_threshold DECIMAL;
    flags TEXT[] := '{}';
    warnings TEXT := '';
    score DECIMAL := 1.0;
    valid BOOLEAN := TRUE;
BEGIN
    SELECT close INTO recent_close
    FROM ig_candles
    WHERE epic = p_epic
      AND timeframe = p_timeframe
      AND start_time < p_candle_time
      AND start_time >= p_candle_time - INTERVAL '1 hour'
    ORDER BY start_time DESC
    LIMIT 1;

    IF recent_close IS NOT NULL THEN
        price_diff_pips := CASE
            WHEN p_epic LIKE '%GOLD%' OR p_epic LIKE '%XAU%' THEN ABS(p_close - recent_close)
            WHEN p_epic LIKE '%SILVER%' OR p_epic LIKE '%XAG%' THEN ABS(p_close - recent_close)
            WHEN p_epic LIKE '%OIL%' OR p_epic LIKE '%BRENT%' OR p_epic LIKE '%WTI%' THEN ABS(p_close - recent_close)
            WHEN p_epic LIKE '%JPY%' THEN ABS(p_close - recent_close) * 100
            ELSE ABS(p_close - recent_close) * 10000
        END;

        effective_threshold := CASE
            WHEN p_epic LIKE '%GOLD%' OR p_epic LIKE '%XAU%' THEN 50.0
            WHEN p_epic LIKE '%SILVER%' OR p_epic LIKE '%XAG%' THEN 50.0  -- was 2.0
            WHEN p_epic LIKE '%OIL%' OR p_epic LIKE '%BRENT%' OR p_epic LIKE '%WTI%' THEN 5.0
            ELSE p_threshold_pips
        END;

        IF price_diff_pips > effective_threshold THEN
            flags := array_append(flags, 'LARGE_PRICE_MOVE');
            warnings := warnings || 'Price moved ' || ROUND(price_diff_pips, 2)::TEXT ||
                CASE
                    WHEN p_epic LIKE '%GOLD%' OR p_epic LIKE '%XAU%' OR p_epic LIKE '%SILVER%' OR p_epic LIKE '%XAG%' OR p_epic LIKE '%OIL%' OR p_epic LIKE '%BRENT%' OR p_epic LIKE '%WTI%'
                    THEN ' USD'
                    ELSE ' pips'
                END || ' from recent candle. ';
            score := score * 0.7;

            IF price_diff_pips > effective_threshold * 3 THEN
                flags := array_append(flags, 'SUSPICIOUS_MOVE');
                warnings := warnings || 'Movement >' || ROUND(effective_threshold * 3, 2)::TEXT ||
                    CASE
                        WHEN p_epic LIKE '%GOLD%' OR p_epic LIKE '%XAU%' OR p_epic LIKE '%SILVER%' OR p_epic LIKE '%XAG%' OR p_epic LIKE '%OIL%' OR p_epic LIKE '%BRENT%' OR p_epic LIKE '%WTI%'
                        THEN ' USD'
                        ELSE ' pips'
                    END || ' flagged as suspicious. ';
                score := score * 0.4;
            END IF;
        END IF;

        IF p_candle_time < CURRENT_TIMESTAMP - INTERVAL '30 minutes' THEN
            flags := array_append(flags, 'STALE_DATA');
            warnings := warnings || 'Data is more than 30 minutes old. ';
            score := score * 0.8;
        END IF;
    END IF;

    IF array_length(flags, 1) > 0 THEN
        INSERT INTO price_validation_log (
            epic, timeframe, candle_time, validation_type, severity, message,
            new_value, price_difference_pips, data_source
        ) VALUES (
            p_epic, p_timeframe, p_candle_time, 'REAL_TIME_VALIDATION',
            CASE WHEN score < 0.3 THEN 'CRITICAL' WHEN score < 0.5 THEN 'WARNING' ELSE 'INFO' END,
            warnings, p_close, price_diff_pips, p_data_source
        );
    END IF;

    valid := score >= 0.3;

    RETURN QUERY SELECT valid, score, flags, warnings;
END;
$$ LANGUAGE plpgsql;
