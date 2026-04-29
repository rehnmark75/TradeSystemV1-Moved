const CACHE_VERSION = "klirr-shell-v1";
const BASE_PATH = "/trading";
const OFFLINE_URL = `${BASE_PATH}/offline.html`;
const APP_SHELL_ASSETS = [
  OFFLINE_URL,
  `${BASE_PATH}/icons/pwa-icon.svg`,
  `${BASE_PATH}/icons/pwa-maskable.svg`
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_VERSION).then((cache) => cache.addAll(APP_SHELL_ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((key) => key !== CACHE_VERSION).map((key) => caches.delete(key)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);

  if (request.method !== "GET") return;
  if (url.origin !== self.location.origin) return;
  if (!url.pathname.startsWith(BASE_PATH)) return;

  if (url.pathname.includes("/api/")) {
    return;
  }

  if (request.mode === "navigate") {
    event.respondWith(
      fetch(request).catch(() => caches.match(OFFLINE_URL))
    );
    return;
  }

  if (
    url.pathname.startsWith(`${BASE_PATH}/_next/static/`) ||
    url.pathname.startsWith(`${BASE_PATH}/icons/`) ||
    url.pathname === OFFLINE_URL
  ) {
    event.respondWith(
      caches.match(request).then((cached) => {
        if (cached) return cached;
        return fetch(request).then((response) => {
          const copy = response.clone();
          caches.open(CACHE_VERSION).then((cache) => cache.put(request, copy));
          return response;
        });
      })
    );
  }
});
