"use client";

import { createContext, useContext, useState, useEffect, ReactNode } from "react";

export type TradingEnvironment = "live" | "demo";

type EnvironmentContextType = {
  environment: TradingEnvironment;
  setEnvironment: (env: TradingEnvironment) => void;
  isLive: boolean;
};

const EnvironmentContext = createContext<EnvironmentContextType>({
  environment: "demo",
  setEnvironment: () => {},
  isLive: false,
});

const STORAGE_KEY = "trading-environment";

export function EnvironmentProvider({ children }: { children: ReactNode }) {
  const [environment, setEnvironmentState] = useState<TradingEnvironment>("demo");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY) as TradingEnvironment | null;
    if (saved === "live" || saved === "demo") {
      setEnvironmentState(saved);
    }
    setMounted(true);
  }, []);

  const setEnvironment = (env: TradingEnvironment) => {
    setEnvironmentState(env);
    localStorage.setItem(STORAGE_KEY, env);
  };

  // Avoid hydration mismatch - render children immediately but with default state
  return (
    <EnvironmentContext.Provider
      value={{
        environment: mounted ? environment : "demo",
        setEnvironment,
        isLive: mounted ? environment === "live" : false,
      }}
    >
      {children}
    </EnvironmentContext.Provider>
  );
}

export function useEnvironment() {
  return useContext(EnvironmentContext);
}
