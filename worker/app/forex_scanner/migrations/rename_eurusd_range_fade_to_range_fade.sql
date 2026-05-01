-- Rename RANGE_FADE strategy DB objects: drop the EURUSD-origin prefix.
-- Strategy now runs across 9 pairs; the "eurusd_" prefix is misleading.
-- Apply atomically together with the matching Python rename PR.

BEGIN;

ALTER TABLE eurusd_range_fade_global_config  RENAME TO range_fade_global_config;
ALTER TABLE eurusd_range_fade_pair_overrides RENAME TO range_fade_pair_overrides;

ALTER INDEX eurusd_range_fade_global_config_pkey
    RENAME TO range_fade_global_config_pkey;
ALTER INDEX eurusd_range_fade_global_config_profile_config_active_key
    RENAME TO range_fade_global_config_profile_config_active_key;
ALTER INDEX eurusd_range_fade_pair_overrides_pkey
    RENAME TO range_fade_pair_overrides_pkey;
ALTER INDEX eurusd_range_fade_pair_overrides_epic_profile_config_key
    RENAME TO range_fade_pair_overrides_epic_profile_config_key;

ALTER SEQUENCE eurusd_range_fade_global_config_id_seq
    RENAME TO range_fade_global_config_id_seq;
ALTER SEQUENCE eurusd_range_fade_pair_overrides_id_seq
    RENAME TO range_fade_pair_overrides_id_seq;

COMMIT;
