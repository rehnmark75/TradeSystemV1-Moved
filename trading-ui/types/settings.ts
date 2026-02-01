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
