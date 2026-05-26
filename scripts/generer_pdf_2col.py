"""
generer_pdf_2col.py
Genererer en to-kolonne preview-PDF av rapport_live.html.
Kjør: python generer_pdf_2col.py
"""

import os
import re
import subprocess
import threading
import time
import http.server
import socketserver

KILDE_HTML = "rapport_live.html"
OUTPUT_PDF  = r"005 report\Prosjekt_LOG650_ME_2kol_forslag.pdf"
TEMP_HTML   = "rapport_print_2col_temp.html"
PORT        = 8767
CHROME      = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

with open(KILDE_HTML, "r", encoding="utf-8") as f:
    html = f.read()

# Fjern auto-refresh
html = re.sub(r'<meta http-equiv="refresh"[^>]*>\s*', "", html)

two_col_css = """
<style>
/* ── To-kolonne akademisk layout ── */
@page { size: A4; margin: 2cm 1.5cm; }

body {
  font-family: "Times New Roman", Times, serif;
  font-size: 9.5pt;
  line-height: 1.5;
  color: #000;
  background: white !important;
  padding: 0 !important;
}

.container {
  max-width: 100% !important;
  padding: 0 !important;
  box-shadow: none !important;
  column-count: 2;
  column-gap: 1.2cm;
  column-rule: 0.3pt solid #ccc;
}

/* Elementer som skal strekke seg over begge kolonner */
.abstract,
.toc {
  column-span: all;
}

figure {
  column-span: all;
  margin: 1em 0;
}

/* Abstract */
.abstract {
  border-top: 1.5pt solid #000;
  border-bottom: 0.5pt solid #000;
  padding: 0.75rem 0;
  margin-bottom: 1rem;
}
.abstract p {
  font-size: 9pt;
  line-height: 1.4;
  margin: 0.3rem 0;
}
.abstract p:first-child {
  font-size: 9pt;
  font-weight: bold;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

/* TOC */
.toc {
  border: 0.5pt solid #ccc;
  padding: 0.5rem 1rem;
  margin-bottom: 1.2rem;
  background: #fafafa;
}
.toc p {
  font-size: 9pt;
  line-height: 1.7;
  margin: 0;
  text-align: left;
}
.toc p:first-child {
  font-weight: bold;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

/* Overskrifter */
h1 {
  font-size: 11pt;
  font-weight: bold;
  margin-top: 1.4rem;
  margin-bottom: 0.3rem;
  break-before: auto;
}
h2 {
  font-size: 10pt;
  font-weight: bold;
  margin-top: 0.9rem;
  margin-bottom: 0.15rem;
}
h3 {
  font-size: 9.5pt;
  font-weight: bold;
  font-style: italic;
  margin-top: 0.7rem;
}

/* Tekst */
p {
  text-align: justify;
  hyphens: auto;
  -webkit-hyphens: auto;
  margin: 0.35rem 0;
  orphans: 3;
  widows: 3;
}

/* Tabeller */
table {
  border-collapse: collapse;
  width: 100%;
  margin: 0.8rem 0;
  font-size: 8.5pt;
  border-top: 1.5pt solid #000;
  border-bottom: 1.5pt solid #000;
}
th, td {
  border: none;
  padding: 0.3rem 0.5rem;
  text-align: left;
}
th {
  font-weight: bold;
  border-bottom: 0.5pt solid #000;
}

/* Figurtekst */
figcaption {
  font-size: 8.5pt;
  text-align: center;
  margin-top: 0.3rem;
  color: #000;
}

/* Sitatbokser (RQ) */
blockquote {
  border: 0.5pt solid #bbb;
  background: #f5f5f5;
  margin: 0.7rem 0;
  padding: 0.45rem 0.8rem;
  font-size: 9pt;
}

/* Kode */
code {
  font-family: "Courier New", monospace;
  font-size: 8.5pt;
  background: #f0f0f0;
  padding: 0 3px;
}

/* Fjern live-indikator */
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

html = html.replace("</head>", two_col_css + "</head>")

with open(TEMP_HTML, "w", encoding="utf-8") as f:
    f.write(html)

# Start lokal HTTP-server
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
print(f"Genererer to-kolonne PDF: {OUTPUT_PDF}")

abs_output = os.path.abspath(OUTPUT_PDF)
url = f"http://localhost:{PORT}/{TEMP_HTML}"

result = subprocess.run([
    CHROME,
    "--headless=new",
    "--disable-gpu",
    "--no-pdf-header-footer",
    "--print-to-pdf=" + abs_output,
    "--virtual-time-budget=10000",
    url,
], capture_output=True, text=True, timeout=60)

httpd.shutdown()
os.remove(TEMP_HTML)

if result.returncode == 0 and os.path.exists(abs_output):
    size_kb = os.path.getsize(abs_output) // 1024
    print(f"PDF generert: {abs_output} ({size_kb} KB)")
else:
    print(f"FEIL (kode {result.returncode})")
    if result.stderr:
        print(result.stderr[:500])
