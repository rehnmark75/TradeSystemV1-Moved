"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import CategoryNav, { type CategoryNavItem } from "../../../components/settings/CategoryNav";
import ScannerToolbar, { type ScannerFilterState } from "../../../components/settings/ScannerToolbar";
import ScannerContextPanel from "../../../components/settings/ScannerContextPanel";
import SettingsField from "../../../components/settings/SettingsField";
import SaveModal from "../../../components/settings/SaveModal";
import ConflictModal from "../../../components/settings/ConflictModal";
import { useScannerConfig } from "../../../hooks/settings/useScannerConfig";
import { useEnvironment } from "../../../lib/environment";
import {
  SCANNER_ICONS,
  SCANNER_LABELS,
  SCANNER_SECTION_ORDER,
  SCANNER_SUBSECTIONS,
} from "../../../lib/settings/scannerSections";
import { getParamRiskLevel } from "../../../lib/settings/riskClassification";
import { logTelemetry } from "../../../lib/settings/telemetry";

function toLabel(value: string) {
  return value.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function matchesQuery(fieldName: string, query: string): boolean {
  if (!query) return true;
  const q = query.toLowerCase();
  return fieldName.toLowerCase().includes(q) || toLabel(fieldName).toLowerCase().includes(q);
}

export default function ScannerSettingsPage() {
  const { environment } = useEnvironment();
  const {
    effectiveData,
    defaults,
    loading,
    error,
    changes,
    updateField,
    saveChanges,
    resetChanges,
    conflict,
    setConflict,
    setChanges,
  } = useScannerConfig(environment);

  const [query, setQuery] = useState("");
  const [filters, setFilters] = useState<ScannerFilterState>({
    modifiedOnly: false,
    criticalOnly: false,
  });
  const [focusedParam, setFocusedParam] = useState<string | null>(null);
  const [activeCategory, setActiveCategory] = useState<string | null>(null);
  const [showSaveModal, setShowSaveModal] = useState(false);

  const sectionRefs = useRef<Map<string, HTMLElement>>(new Map());

  useEffect(() => {
    logTelemetry({ event_type: "page_view", page: "/settings/scanner" });
  }, []);

  const config = effectiveData as Record<string, unknown> | null;

  // Build visible subsections per category based on search + filters
  const visibleSections = useMemo(() => {
    if (!config) return [];

    return SCANNER_SECTION_ORDER.map((categoryKey) => {
      const subsections = SCANNER_SUBSECTIONS[categoryKey] ?? [];

      const visibleSubsections = subsections
        .map((sub) => {
          const visibleFields = sub.fields.filter((field) => {
            // Field must exist in config
            if (!(field in config) && !(field in defaults)) return false;

            // Search filter
            if (!matchesQuery(field, query)) return false;

            // Modified filter
            if (filters.modifiedOnly && !(field in changes)) return false;

            // Critical filter
            if (filters.criticalOnly && getParamRiskLevel(field) === "normal") return false;

            return true;
          });

          return { ...sub, visibleFields };
        })
        .filter((sub) => sub.visibleFields.length > 0);

      return { categoryKey, subsections: visibleSubsections };
    }).filter((cat) => cat.subsections.length > 0);
  }, [config, defaults, query, filters, changes]);

  // CategoryNav items
  const categoryNavItems: CategoryNavItem[] = useMemo(() => {
    return SCANNER_SECTION_ORDER.map((categoryKey) => {
      const subsections = SCANNER_SUBSECTIONS[categoryKey] ?? [];
      const allFields = subsections.flatMap((s) => s.fields);
      const modifiedCount = allFields.filter((f) => f in changes).length;
      return {
        category: categoryKey,
        fieldCount: allFields.length,
        modifiedCount,
        overriddenCount: 0,
        displayLabel: SCANNER_LABELS[categoryKey],
        displayIcon: SCANNER_ICONS[categoryKey],
      };
    });
  }, [changes]);

  const pendingCount = Object.keys(changes).length;

  // IntersectionObserver for active category tracking
  useEffect(() => {
    const refs = sectionRefs.current;
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setActiveCategory(entry.target.getAttribute("data-category"));
            break;
          }
        }
      },
      { threshold: 0.1, rootMargin: "-100px 0px -60% 0px" }
    );
    refs.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, [visibleSections]);

  const scrollToCategory = (category: string) => {
    const el = sectionRefs.current.get(category);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const handleSave = () => setShowSaveModal(true);

  const handleConfirmSave = async (reason: string) => {
    await saveChanges({ updatedBy: "admin", changeReason: reason });
    setShowSaveModal(false);
  };

  const handleDiscard = () => resetChanges();

  const renderField = (field: string) => {
    const value = config?.[field];
    const defaultValue = defaults[field];
    const pending = field in changes;
    const riskLevel = getParamRiskLevel(field);

    // Infer type from value or default
    const rawVal = value ?? defaultValue;
    const isBoolean = typeof rawVal === "boolean";
    const isNumber = typeof rawVal === "number";

    // Build a minimal metadata-like object for SettingsField
    const dataType = isBoolean ? "boolean" : isNumber ? "decimal" : "text";
    const defaultStr = defaultValue !== undefined && defaultValue !== null
      ? String(defaultValue)
      : null;

    return (
      <SettingsField
        key={field}
        name={field}
        label={toLabel(field)}
        value={value}
        defaultValue={defaultValue}
        pending={pending}
        metadata={{
          id: 0,
          parameter_name: field,
          display_name: toLabel(field),
          category: "",
          data_type: dataType,
          display_order: 0,
          default_value: defaultStr,
          min_value: null,
          max_value: null,
          unit: null,
          valid_options: null,
          help_text: null,
          description: null,
          requires_restart: false,
          is_advanced: false,
        }}
        onFocus={() => setFocusedParam(field)}
        onChange={(v) => updateField(field, v)}
      />
    );
  };

  const originalValues = useMemo(() => config ?? {}, [config]);

  return (
    <div className="strategy-page">
      <ScannerToolbar
        query={query}
        onQueryChange={setQuery}
        filters={filters}
        onFiltersChange={setFilters}
        pendingCount={pendingCount}
        onSave={handleSave}
        onDiscard={handleDiscard}
      />

      <div className="strategy-layout">
        <CategoryNav
          items={categoryNavItems}
          activeCategory={activeCategory ?? undefined}
          onSelect={scrollToCategory}
        />

        <div className="strategy-content">
          {loading ? (
            <div className="settings-placeholder">Loading scanner settings…</div>
          ) : error ? (
            <div className="settings-placeholder">Error: {error}</div>
          ) : visibleSections.length === 0 ? (
            <div className="settings-placeholder">No parameters match your search.</div>
          ) : (
            visibleSections.map(({ categoryKey, subsections }) => {
              const totalFields = subsections.reduce((n, s) => n + s.visibleFields.length, 0);
              return (
                <section
                  key={categoryKey}
                  className="strategy-section"
                  data-category={categoryKey}
                  ref={(el) => {
                    if (el) sectionRefs.current.set(categoryKey, el);
                    else sectionRefs.current.delete(categoryKey);
                  }}
                >
                  <div className="strategy-section-header">
                    <h2>
                      <span className="scanner-section-icon">{SCANNER_ICONS[categoryKey]}</span>
                      {SCANNER_LABELS[categoryKey]}
                    </h2>
                    <span className="strategy-section-count">{totalFields} fields</span>
                  </div>

                  {subsections.map((sub) => (
                    <div key={sub.key} className="strategy-subgroup">
                      <div className="strategy-subgroup-title">
                        {sub.title}
                        {sub.critical ? (
                          <span className="subgroup-critical-badge" title="Critical settings">⚠</span>
                        ) : null}
                      </div>
                      {sub.description ? (
                        <div className="strategy-subgroup-desc">{sub.description}</div>
                      ) : null}
                      <div className="strategy-field-grid">
                        {sub.visibleFields.map(renderField)}
                      </div>
                    </div>
                  ))}
                </section>
              );
            })
          )}
        </div>

        <ScannerContextPanel
          paramName={focusedParam}
          currentValue={focusedParam ? config?.[focusedParam] : undefined}
          defaultValue={focusedParam ? defaults[focusedParam] : undefined}
          onReset={
            focusedParam
              ? () => {
                  const def = defaults[focusedParam!];
                  if (def !== undefined) updateField(focusedParam!, def);
                }
              : undefined
          }
        />
      </div>

      {showSaveModal ? (
        <SaveModal
          changes={changes as Record<string, unknown>}
          originalValues={originalValues as Record<string, unknown>}
          onConfirm={handleConfirmSave}
          onCancel={() => setShowSaveModal(false)}
        />
      ) : null}

      <ConflictModal
        open={!!conflict}
        current={conflict}
        pending={changes}
        onClose={() => setConflict(null)}
        onResolve={async ({ action, mergedChanges }) => {
          if (!conflict) return;
          if (action === "discard") { resetChanges(); setConflict(null); return; }
          if (mergedChanges) {
            setChanges(mergedChanges);
            await saveChanges(
              { updatedBy: "admin", changeReason: "Conflict resolution" },
              mergedChanges,
              (conflict as Record<string, unknown>).updated_at as string
            );
            setConflict(null);
          }
        }}
      />
    </div>
  );
}
