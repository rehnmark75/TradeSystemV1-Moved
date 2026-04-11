export interface SettingsObject {
  [key: string]: SettingsValue;
}

export type SettingsValue =
  | string
  | number
  | boolean
  | null
  | SettingsValue[]
  | SettingsObject;

export type ScannerConfig = {
  id: number;
  version: string;
  is_active?: boolean | null;
  updated_at: string;
  updated_by?: string | null;
  change_reason?: string | null;
} & Record<string, SettingsValue>;

export type SmcSimpleConfig = {
  id: number;
  version: string;
  is_active?: boolean | null;
  updated_at: string;
  updated_by?: string | null;
  change_reason?: string | null;
} & Record<string, SettingsValue>;

export interface PairOverride {
  id: number;
  epic: string;
  config_id: number | null;
  overrides: Record<string, SettingsValue>;
  updated_at: string;
  updated_by?: string | null;
  change_reason?: string | null;
}

export interface AuditEntry {
  id: number;
  config_id: number;
  change_type: string;
  changed_by: string;
  changed_at: string;
  change_reason?: string | null;
  previous_values?: Record<string, SettingsValue> | null;
  new_values?: Record<string, SettingsValue> | null;
  category?: string | null;
}

export interface ConfigSnapshot {
  id: number;
  snapshot_name: string;
  description?: string | null;
  base_config_id?: number | null;
  base_config_version?: string | null;
  parameter_overrides: Record<string, SettingsValue>;
  created_at: string;
  updated_at: string;
  created_by?: string | null;
  last_tested_at?: string | null;
  test_results?: Record<string, SettingsValue> | null;
  test_count: number;
  is_promoted: boolean;
  is_backtest_only: boolean;
  is_active: boolean;
  tags: string[];
}

export interface SnapshotDiff {
  field: string;
  snapshot_value: SettingsValue;
  current_value: SettingsValue;
  changed: boolean;
}

export interface SmcParameterMetadata {
  id: number;
  parameter_name: string;
  display_name: string;
  category: string;
  subcategory?: string | null;
  data_type: string;
  min_value?: string | number | null;
  max_value?: string | number | null;
  default_value?: string | null;
  description?: string | null;
  help_text?: string | null;
  display_order?: number | null;
  is_advanced?: boolean | null;
  requires_restart?: boolean | null;
  valid_options?: unknown | null;
  unit?: string | null;
}
