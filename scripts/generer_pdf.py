"""
generer_pdf.py
Genererer PDF av rapport_live.html via Chrome headless — to-kolonne akademisk layout.
Kjør: python generer_pdf.py
"""

import os
import re
import subprocess
import threading
import time
import http.server
import socketserver

# --- Konfigurasjon ---
KILDE_HTML = "rapport_live.html"
OUTPUT_PDF  = r"005 report\Prosjekt_LOG650_ME_Morten_Eidsvag.pdf"
TEMP_HTML   = "rapport_print_temp.html"
PORT        = 8766
CHROME      = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

# --- Lag print-versjon av HTML ---
with open(KILDE_HTML, "r", encoding="utf-8") as f:
    html = f.read()

# Fjern auto-refresh
html = re.sub(r'<meta http-equiv="refresh"[^>]*>\s*', "", html)

print_css = """
<style>
/* ════════════════════════════════════════════
   To-kolonne akademisk layout — LOG650
   ════════════════════════════════════════════ */

@page {
  size: A4;
  margin: 2cm 1.5cm 2.2cm 1.5cm;
}

/* ── Grunnleggende typografi ── */
body {
  font-family: "Times New Roman", Times, serif;
  font-size: 9.5pt;
  line-height: 1.5;
  color: #000;
  background: white !important;
  padding: 0 !important;
  margin: 0 !important;
}

/* ── Hoved-container: to kolonner ── */
.container {
  max-width: 100% !important;
  padding: 0 !important;
  box-shadow: none !important;
  column-count: 2;
  column-gap: 1.1cm;
  column-rule: 0.3pt solid #ccc;
}

/* ── Elementer som strekker seg over begge kolonner ── */
.abstract,
.toc,
figure {
  column-span: all;
}

/* Tabeller flyter innenfor kolonnen */
table {
  column-span: none;
  width: 100%;
  font-size: 7.5pt;
}

/* ── Tittelseksjon ── */
.title-section {
  column-span: all;
  text-align: center;
  padding: 1.2rem 0 1rem 0;
  margin-bottom: 0.8rem;
}
.paper-title {
  font-size: 15pt;
  font-weight: bold;
  line-height: 1.3;
  margin: 0 0 0.6rem 0;
  text-align: center;
}
.paper-subtitle {
  font-size: 10.5pt;
  font-style: italic;
  margin: 0 0 0.9rem 0;
  line-height: 1.4;
  text-align: center;
}
.paper-author {
  font-size: 11pt;
  font-weight: bold;
  margin: 0 0 0.25rem 0;
  text-align: center;
}
.paper-meta {
  font-size: 9pt;
  color: #333;
  margin: 0;
  text-align: center;
}

/* ── Abstract ── */
.abstract {
  border-top: 1.5pt solid #000;
  border-bottom: 0.5pt solid #000;
  padding: 0.8rem 0;
  margin-bottom: 1.1rem;
}
.abstract p {
  font-size: 9pt;
  line-height: 1.45;
  margin: 0.3rem 0;
  text-align: justify;
  hyphens: auto;
}

/* ── Innholdsfortegnelse ── */
.toc {
  border: 0.5pt solid #ccc;
  padding: 0.6rem 1rem;
  margin-bottom: 1.2rem;
  background: #fafafa;
}
.toc p {
  font-size: 9pt;
  line-height: 1.8;
  margin: 0;
  text-align: left;
}
.toc p:first-child {
  font-weight: bold;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-bottom: 0.35rem;
}

/* ── Overskrifter ── */
h1 {
  font-size: 11pt;
  font-weight: bold;
  margin-top: 1.4rem;
  margin-bottom: 0.35rem;
  break-after: avoid;
}
h2 {
  font-size: 10pt;
  font-weight: bold;
  margin-top: 1rem;
  margin-bottom: 0.15rem;
  break-after: avoid;
}
h3 {
  font-size: 9.5pt;
  font-weight: bold;
  font-style: italic;
  margin-top: 0.8rem;
  margin-bottom: 0.1rem;
  break-after: avoid;
}

/* ── Brødtekst ── */
p {
  text-align: justify;
  hyphens: auto;
  -webkit-hyphens: auto;
  margin: 0.4rem 0;
  orphans: 3;
  widows: 3;
}

/* ── Tabeller (strekker seg over begge kolonner) ── */
table {
  border-collapse: collapse;
  width: 100%;
  margin: 1rem 0;
  font-size: 8.5pt;
  border-top: 1.5pt solid #000;
  border-bottom: 1.5pt solid #000;
  break-inside: avoid;
}
th, td {
  border: none;
  padding: 0.3rem 0.6rem;
  text-align: left;
}
th {
  font-weight: bold;
  border-bottom: 0.5pt solid #000;
}
tr:nth-child(even) { background: none; }

/* ── Figurer (strekker seg over begge kolonner) ── */
figure {
  margin: 1.2em 0;
  break-inside: avoid;
}
img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 0 auto;
}
figcaption {
  font-size: 8.5pt;
  text-align: center;
  font-style: normal;
  margin-top: 0.4rem;
  color: #000;
}

/* ── Matematiske formler (innenfor kolonnen, skalert ned) ── */
mjx-container[display="true"] {
  font-size: 78% !important;
  margin: 0.5rem 0;
  max-width: 100%;
  overflow-x: hidden;
  display: block;
}
mjx-container[display="true"] > svg,
mjx-container[display="true"] > mjx-math {
  max-width: 100%;
}
mjx-container:not([display="true"]) {
  font-size: 90% !important;
}
.math.display {
  font-size: 78%;
  margin: 0.5rem 0;
  max-width: 100%;
  overflow-x: hidden;
}

/* ── Sitatbokser (forskningsspørsmål) ── */
blockquote {
  border: 0.5pt solid #bbb;
  background: #f5f5f5;
  margin: 0.8rem 0;
  padding: 0.5rem 0.85rem;
  font-size: 9pt;
  break-inside: avoid;
}

/* ── Kode ── */
code {
  font-family: "Courier New", monospace;
  font-size: 8.5pt;
  background: #f0f0f0;
  padding: 0 3px;
  border-radius: 2px;
}

/* ── Lister ── */
ul, ol {
  margin: 0.4rem 0 0.4rem 1.2rem;
  padding: 0;
}
li {
  margin: 0.15rem 0;
  font-size: 9.5pt;
}

/* ── Fjern live-indikator ── */
body::after { display: none !important; }
</style>

<script>
  window.mjReady = false;
  document.addEventListener('DOMContentLoaded', function() {
    if (window.MathJax && window.MathJax.startup) {
      window.MathJax.startup.promise.then(function() { window.mjReady = true; });
    } else {
      window.mjReady = true;
    }
  });
</script>
"""

html = html.replace("</head>", print_css + "</head>")

with open(TEMP_HTML, "w", encoding="utf-8") as f:
    f.write(html)

# --- Start lokal HTTP-server ---
class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

httpd = socketserver.TCPServer(("", PORT), QuietHandler)
server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
server_thread.start()

for _ in range(20):
    try:
        import socket
        s = socket.create_connection(("localhost", PORT), timeout=0.5)
        s.close()
        break
    except OSError:
        time.sleep(0.3)

print(f"Server klar på http://localhost:{PORT}")
print(f"Genererer PDF: {OUTPUT_PDF}")

abs_output = os.path.abspath(OUTPUT_PDF)
url = f"http://localhost:{PORT}/{TEMP_HTML}"

result = subprocess.run([
    CHROME,
    "--headless=new",
    "--disable-gpu",
    "--no-pdf-header-footer",
    "--print-to-pdf=" + abs_output,
    "--virtual-time-budget=12000",
    url,
], capture_output=True, text=True, timeout=60)

# --- Rydd opp ---
httpd.shutdown()
os.remove(TEMP_HTML)

# --- Resultat ---
if result.returncode == 0 and os.path.exists(abs_output):
    size_kb = os.path.getsize(abs_output) // 1024
    print(f"PDF generert: {abs_output} ({size_kb} KB)")
else:
    print(f"FEIL (kode {result.returncode})")
    if result.stderr:
        print(result.stderr[:500])
