const offlineHtml = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#0b1728" />
    <title>K.L.I.R.R Offline</title>
    <style>
      :root{color-scheme:dark;--bg:#07111f;--panel:#0b1728;--line:rgba(159,185,227,.18);--ink:#f4f7fb;--muted:#a9b8cc;--accent:#7cc7ff}
      *{box-sizing:border-box}
      body{margin:0;min-height:100dvh;display:grid;place-items:center;padding:22px;font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--ink);background:radial-gradient(circle at top left,rgba(55,92,160,.35),transparent 30%),linear-gradient(180deg,#081120 0%,var(--bg) 100%)}
      main{width:min(460px,100%);border:1px solid var(--line);border-radius:22px;padding:24px;background:linear-gradient(180deg,rgba(15,29,49,.94),rgba(10,22,38,.94));box-shadow:0 30px 70px rgba(0,0,0,.34)}
      .mark{margin:0 0 18px;font-weight:800;letter-spacing:-.03em;font-size:1.45rem}
      h1{margin:0 0 10px;font-size:1.7rem;line-height:1.05}
      p{margin:0;color:var(--muted);line-height:1.55}
      a{display:inline-flex;margin-top:20px;color:var(--accent);text-decoration:none;font-weight:700}
    </style>
  </head>
  <body>
    <main>
      <p class="mark">K.L.I.R.R</p>
      <h1>Connection required</h1>
      <p>The app shell is available, but live market data, signals, and trading actions require a network connection.</p>
      <a href="/trading/">Try again</a>
    </main>
  </body>
</html>`;

export function GET() {
  return new Response(offlineHtml, {
    headers: {
      "Content-Type": "text/html; charset=utf-8",
      "Cache-Control": "public, max-age=3600"
    }
  });
}
