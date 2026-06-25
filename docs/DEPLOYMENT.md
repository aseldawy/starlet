# Deploying a Starlet tile server

This is a step-by-step guide to standing up a live Starlet vector-tile server,
from a raw GeoParquet/GeoJSON file to tiles served over HTTP — including a
**no-root** recipe for running behind an existing Apache install (the exact
setup used for the `starmap.cs.ucr.edu/demo` deployment).

- [1. Prerequisites](#1-prerequisites)
- [2. Install Starlet](#2-install-starlet)
- [3. Build a dataset](#3-build-a-dataset)
- [4. Run the server](#4-run-the-server)
- [5. Expose it behind Apache without root (CGI bridge)](#5-expose-it-behind-apache-without-root-cgi-bridge)
- [6. Keep it running (auto-start, no root)](#6-keep-it-running-auto-start-no-root)
- [7. (Optional) A web map viewer](#7-optional-a-web-map-viewer)
- [8. Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

- **Python 3.10+** (`python3.12` recommended).
- Disk space for the dataset and its tiles. Put everything on a roomy volume
  (e.g. `/data`) if your home/`/var/www` partition is small.
- For the no-root Apache deployment (§5): an Apache that already serves your
  site, with `mod_cgi`/`mod_cgid` loaded and `.htaccess` overrides allowed
  (`Options`, `AddHandler`). You do **not** need `mod_wsgi` or root.

Check Python:

```bash
python3.12 --version          # Python 3.12.x
```

## 2. Install Starlet

Create an isolated virtualenv (on your data disk if space is tight) and install
the package plus its geo stack:

```bash
# pick a location with space; avoid a near-full home/root partition
mkdir -p /data/$USER/starlet_demo && cd /data/$USER/starlet_demo

python3.12 -m venv venv
./venv/bin/python -m pip install --upgrade pip wheel
./venv/bin/pip install starlet            # pulls pyarrow, geopandas, shapely, pyproj, flask, pmtiles, ...

./venv/bin/starlet --version              # 0.3.0
```

> Installing from a local wheel instead of PyPI:
> `./venv/bin/pip install ./starlet-0.3.0-py3-none-any.whl`

## 3. Build a dataset

`starlet build` runs the full pipeline — partition into GeoParquet shards
(`tile`) and pre-generate the MVT pyramid (`mvt`). A *dataset* is a directory
under an output root; the server serves every dataset directory it finds.

```bash
mkdir -p datasets

./venv/bin/starlet build \
  --input  /path/to/source.parquet \
  --outdir datasets/MyDataset \
  --zoom   8 \
  --threshold 1            # generate ALL non-empty tiles at every zoom (see note)
```

Result layout:

```
datasets/MyDataset/
  parquet_tiles/      # spatial shards (the index + a queryable GeoParquet store)
  histograms/         # density grid for tile selection
  stats/attributes.json
  mvt/<z>/<x>/<y>.mvt  # the pre-generated pyramid
```

**Notes & gotchas**

- **Geometry column.** If your geometry column is not named `geometry`
  (e.g. OGR/`pyogrio` exports often use `wkb_geometry`), pass
  `--geom-col wkb_geometry`, or rename it first. A wrong name yields
  *"No geometries sampled to build RSGrove index."*
- **Build a *complete* pyramid for a smooth viewer.** Use a **low**
  `--threshold` (e.g. `1`) so every non-empty tile at every zoom is generated.
  A high threshold pre-generates only dense tiles and leaves gaps; the viewer
  then shows blank areas at intermediate zooms.
- **GeoJSON input** works too (`--input source.geojson`). For very large GeoJSON,
  convert to GeoParquet first (e.g. via `pyogrio`/GDAL with
  `OGR_GEOJSON_MAX_OBJ_SIZE=0` for oversized features).
- Tile-only (no pyramid): run `starlet tile ...` then serve — tiles are then
  produced on demand. Pre-generating (above) is recommended for a snappy demo.
- Pack the pyramid into a single archive for distribution:
  `starlet build ... --pmtiles` writes `datasets/MyDataset.pmtiles`.

## 4. Run the server

```bash
./venv/bin/starlet serve \
  --dir  datasets \
  --host 127.0.0.1 \
  --port 8765 \
  --log-level WARNING
```

Endpoints (relative to `http://127.0.0.1:8765`):

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/datasets` | List datasets |
| GET | `/api/datasets/<d>/stats` | Attribute statistics |
| GET | `/datasets/<d>.json` | Dataset metadata (size, geometry, …) |
| GET | `/<d>/<z>/<x>/<y>.mvt` | Vector tile |
| GET/POST | `/datasets/<d>/features.<csv\|geojson>?mbr=…` | Bounded feature download |

Smoke test:

```bash
curl -s http://127.0.0.1:8765/api/datasets
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8765/MyDataset/0/0/0.mvt
```

> Bind to `127.0.0.1` so only the local Apache (next section) can reach it. Use
> `--host 0.0.0.0` only if you intend to expose the port directly.

> **Restart after changing datasets.** The server builds its spatial index
> (`ParquetIndex`) once, when a dataset is first requested, and caches it. If you
> add a dataset or rebuild a dataset's `parquet_tiles/` while the server is
> running, **restart `serve`** (or re-run `serve_start.sh` after `pkill -f
> "starlet serve"`) so the new files are picked up — otherwise on-the-fly tiles
> for the changed dataset come back empty.

## 5. Expose it behind Apache without root (CGI bridge)

`mod_wsgi` and `ProxyPass` both require editing the Apache vhost (root). If you
only control your site directory, bridge to the running Flask process with a
tiny **CGI reverse proxy** — allowed from `.htaccess` via `mod_cgi`.

Create `proxy.cgi` in your web directory (e.g. `…/public_html/demo/proxy.cgi`):

```python
#!/usr/bin/python3
import os, sys, urllib.request, urllib.error
BACKEND = "http://127.0.0.1:8765"
path = os.environ.get("PATH_INFO", "/")
qs   = os.environ.get("QUERY_STRING", "")
url  = BACKEND + path + (("?" + qs) if qs else "")
method = os.environ.get("REQUEST_METHOD", "GET")
data = None
if method == "POST":
    n = int(os.environ.get("CONTENT_LENGTH") or 0)
    data = sys.stdin.buffer.read(n) if n else b""
req = urllib.request.Request(url, data=data, method=method)
ct = os.environ.get("CONTENT_TYPE")
if ct: req.add_header("Content-Type", ct)
try:
    r = urllib.request.urlopen(req, timeout=120)
    sys.stdout.write("Content-Type: %s\r\n" % r.headers.get("Content-Type", "application/octet-stream"))
    sys.stdout.write("Access-Control-Allow-Origin: *\r\n\r\n"); sys.stdout.flush()
    sys.stdout.buffer.write(r.read())
except urllib.error.HTTPError as e:
    sys.stdout.write("Status: %d\r\nContent-Type: text/plain\r\n\r\n" % e.code); sys.stdout.flush()
    sys.stdout.buffer.write(e.read())
except Exception as e:
    sys.stdout.write("Status: 502\r\nContent-Type: text/plain\r\n\r\nproxy error: %s" % e)
```

Enable CGI in the same directory's `.htaccess`:

```apache
Options +ExecCGI +FollowSymLinks
AddHandler cgi-script .cgi
```

```bash
chmod 755 .../public_html/demo/proxy.cgi
```

Now every backend path is reachable publicly via `PATH_INFO`:

```bash
curl https://your.site/demo/proxy.cgi/api/datasets
curl -o /dev/null -w "%{http_code}\n" https://your.site/demo/proxy.cgi/MyDataset/0/0/0.mvt
```

> The shim only imports `urllib` (no heavy deps), so per-request CGI startup is
> a few ms; the actual tile work happens in the persistent Flask process.

## 6. Keep it running (auto-start, no root)

Use a user `cron` to start the server on boot and restart it if it dies — no
systemd/root needed.

`serve_start.sh`:

```bash
#!/usr/bin/env bash
cd /data/$USER/starlet_demo
# already up? then do nothing
if curl -s -o /dev/null --max-time 5 http://127.0.0.1:8765/api/datasets; then exit 0; fi
nohup ./venv/bin/starlet serve --dir datasets --host 127.0.0.1 --port 8765 \
  --log-level WARNING >> serve.log 2>&1 &
```

```bash
chmod +x serve_start.sh
( crontab -l 2>/dev/null | grep -v serve_start.sh
  echo "@reboot /data/$USER/starlet_demo/serve_start.sh"
  echo "*/5 * * * * /data/$USER/starlet_demo/serve_start.sh" ) | crontab -
```

## 7. (Optional) A web map viewer

Any MapLibre/OpenLayers page can consume the tiles. Point the source at the CGI
bridge and use **absolute** tile URLs (MapLibre's worker can't resolve
root-relative ones), and the layer name **`layer0`** (Starlet's MVT layer):

```js
const BASE = window.location.origin + '/demo/proxy.cgi';
map.addSource('src', {
  type: 'vector',
  tiles: [BASE + '/MyDataset/{z}/{x}/{y}.mvt'],
  minzoom: 0,
  maxzoom: 8            // == your pre-generated max zoom; deeper views overzoom
});
map.addLayer({
  id: 'fill', type: 'fill', source: 'src',
  'source-layer': 'layer0',
  paint: { 'fill-color': '#3b82f6', 'fill-opacity': 0.45 }
});
```

## 8. Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `No geometries sampled to build RSGrove index` | Wrong geometry column — pass `--geom-col wkb_geometry` (or rename to `geometry`). |
| Map shows only the basemap; tiles are empty (`15 b`, 0 features) at mid zooms | Either a sparse pyramid (rebuild MVTs with a low `--threshold`, e.g. `1`) **or** a stale index: the server built its `ParquetIndex` before `parquet_tiles/` was fully in place. **Restart `serve`** after the dataset directory is complete. |
| On-the-fly tiles (beyond the pre-generated zooms) are empty even though `parquet_tiles/` exist | Stale `ParquetIndex` — restart `serve` (the index is built once at startup; files added afterward aren't seen). |
| Blank beyond a certain zoom | Viewer source `maxzoom` is higher than the pre-generated pyramid. Set `maxzoom` to the pyramid's max so MapLibre overzooms instead of requesting missing tiles. |
| `Failed to construct 'Request': Failed to parse URL …/{z}/{x}/{y}.mvt` | Tile URL is relative — make it absolute (`window.location.origin + …`). |
| Browser shows stale behavior after an edit | Bump a cache-buster on JS/CSS (`view.js?v=N`) or hard-reload. |
| `proxy error:` in responses | The Flask backend is down — check `serve.log`, re-run `serve_start.sh`. |
| 404 on `/api/datasets/<d>/stats` | The dataset is missing `stats/attributes.json` (e.g. an incomplete copy). Re-transfer/rebuild the dataset directory. |
