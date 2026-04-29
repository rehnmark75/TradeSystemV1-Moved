"use client";

import { useEffect } from "react";

const BASE_PATH = "/trading";

export default function PwaRegistration() {
  useEffect(() => {
    if (!("serviceWorker" in navigator)) return;
    if (process.env.NODE_ENV !== "production") return;

    window.addEventListener("load", () => {
      navigator.serviceWorker.register(`${BASE_PATH}/sw.js`, {
        scope: `${BASE_PATH}/`
      }).catch((error) => {
        console.warn("PWA service worker registration failed", error);
      });
    });
  }, []);

  return null;
}
