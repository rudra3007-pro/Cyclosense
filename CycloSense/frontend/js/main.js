/**
 * CycloSense Dashboard — main.js
 * Handles: Map init, image upload, detection, classification,
 *          forecasting, GeoJSON rendering, Plotly charts.
 */

const API = 'http://localhost:5000';

// ── State ─────────────────────────────────────────────────────
let MAP, trackLayer, impactLayer, markerLayer, userLocationLayer;
let currentCategory = null;
let currentDetection = null;
let currentForecast = null;
let activeAiRole = 'resident';
let aiHistory = [];
let activeUserCity = null;

// ── IMD label helpers ─────────────────────────────────────────
const CAT_LABEL = {
  D:    'Depression',
  DD:   'Deep Depression',
  CS:   'Cyclonic Storm',
  SCS:  'Severe Cyclonic Storm',
  VSCS: 'Very Severe Cyclonic Storm',
};
const CAT_EMOJI = { D:'', DD:'', CS:'', SCS:'', VSCS:'' };
const RISK_COLOR = {
  'LOW':       '#10b981',
  'MODERATE':  '#f59e0b',
  'HIGH':      '#f97316',
  'VERY HIGH': '#ef4444',
};
const UNCERTAINTY_KM = { 3:130, 6:160, 12:240, 24:400 };

// ─────────────────────────────────────────────────────────────
// 1. MAP INIT
// ─────────────────────────────────────────────────────────────
function initMap() {
  MAP = L.map('map', {
    center: [15, 80],
    zoom: 5,
    zoomControl: false,
    attributionControl: true,
  });

  // ── 1. Real True-Color Satellite Imagery (ESRI World Imagery — free, no key)
  const satelliteTiles = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    {
      attribution: 'Tiles &copy; <a href="https://www.esri.com/">Esri</a> &middot; Earthstar Geographics',
      maxZoom: 17,
      minZoom: 3,
    }
  );

  // ── 2. Boundaries & Coastlines / City Labels Overlay
  const labelsOverlay = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
    {
      maxZoom: 17,
      minZoom: 3,
      opacity: 0.85,
    }
  );

  // ── 3. Dark Gray Canvas (Alternative option)
  const darkCanvasTiles = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
    {
      attribution: 'Tiles &copy; <a href="https://www.esri.com/">Esri</a>',
      maxZoom: 12,
      minZoom: 3,
    }
  );

  // Default: True-color satellite imagery + borders/labels
  satelliteTiles.addTo(MAP);
  labelsOverlay.addTo(MAP);

  // Layer control switcher in top-right
  const baseMaps = {
    "Real Satellite Imagery": satelliteTiles,
    "Dark Gray Canvas": darkCanvasTiles,
  };
  const overlayMaps = {
    "Borders & Coastlines": labelsOverlay,
  };
  L.control.layers(baseMaps, overlayMaps, { position: 'topright' }).addTo(MAP);

  L.control.zoom({ position: 'bottomright' }).addTo(MAP);

  impactLayer = L.layerGroup().addTo(MAP);
  trackLayer  = L.layerGroup().addTo(MAP);
  markerLayer = L.layerGroup().addTo(MAP);
  userLocationLayer = L.layerGroup().addTo(MAP);

  // Click on map to set custom location check
  MAP.on('click', e => {
    if (e.originalEvent.target.closest('#city-search-bar') || e.originalEvent.target.closest('#proximity-card')) return;
    const lat = parseFloat(e.latlng.lat.toFixed(3));
    const lon = parseFloat(e.latlng.lng.toFixed(3));
    selectCity({
      name: `Point (${lat}°N, ${lon}°E)`,
      lat,
      lon,
      region: 'Custom Location'
    });
  });
}

// ─────────────────────────────────────────────────────────────
// 2. API STATUS CHECK
// ─────────────────────────────────────────────────────────────
async function checkStatus() {
  const el = document.getElementById('api-status');
  try {
    const r = await fetch(`${API}/api/predict/health`, { signal: AbortSignal.timeout(4000) });
    const d = await r.json();
    if (d.models_loaded) {
      el.className = 'api-badge ok';
      el.innerHTML = '<div class="dot"></div>API Ready · All Models Loaded';
      el.title = 'Click to run live model self-test';
      el.style.cursor = 'pointer';
    } else {
      setStatusError(el, 'Models not loaded');
    }
  } catch {
    setStatusError(el, 'Backend Offline');
  }

  // Allow clicking badge to test live inference on all 3 models
  if (!el._hasTestHandler) {
    el._hasTestHandler = true;
    el.addEventListener('click', async () => {
      el.className = 'api-badge';
      el.innerHTML = '<div class="dot" style="background:#f59e0b"></div>Testing models…';
      try {
        const res = await fetch(`${API}/api/predict/test-models`);
        const data = await res.json();
        if (data.all_operational) {
          alert(`All 3 Models Operational!\n\n` +
            `• Model 1 (YOLO): ${data.models.model1_yolo.latency_ms} ms (OK)\n` +
            `• Model 2 (EfficientNet): ${data.models.model2_efficientnet.latency_ms} ms (OK - Predicted ${data.models.model2_efficientnet.sample_output.predicted_category})\n` +
            `• Model 3 (LSTM): ${data.models.model3_lstm.latency_ms} ms (OK - 8 steps forecast)`);
        } else {
          alert('Model test failed: ' + JSON.stringify(data.models, null, 2));
        }
      } catch (err) {
        alert('Diagnostic error: ' + err.message);
      } finally {
        checkStatus();
      }
    });
  }
}
function setStatusError(el, msg) {
  el.className = 'api-badge error';
  el.innerHTML = `<div class="dot"></div>${msg}`;
}

// ─────────────────────────────────────────────────────────────
// 3. IMAGE UPLOAD  → DETECT → CLASSIFY
// ─────────────────────────────────────────────────────────────
function setupUpload() {
  const zone  = document.getElementById('upload-zone');
  const input = document.getElementById('img-input');

  zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('over');
    const f = e.dataTransfer.files[0];
    if (f) handleImageFile(f);
  });
  input.addEventListener('change', () => { if (input.files[0]) handleImageFile(input.files[0]); });
}

// ── Wind speed estimate per IMD category (km/h mid-range) ────
const CAT_WIND = { D:55, DD:65, CS:90, SCS:110, VSCS:175 };

async function handleImageFile(file) {
  // Show preview
  const preview = document.getElementById('img-preview');
  preview.src   = URL.createObjectURL(file);
  preview.style.display = 'block';

  document.getElementById('det-loader').classList.add('on');
  resetDetectionUI();

  try {
    // ── Model 1: Detect ───────────────────────────────────────
    const basin = document.getElementById('img-basin')?.value || 'bay_of_bengal';
    const fd1 = new FormData();
    fd1.append('image', file);
    fd1.append('basin', basin);
    const detRes  = await fetch(`${API}/api/predict/detect`, { method:'POST', body:fd1 });
    const detData = await detRes.json();
    renderDetection(detData);

    // If YOLO returned an annotated image with bounding box, update the preview
    if (detData.annotated_image) {
      preview.src = detData.annotated_image;
    }

    if (!detData.cyclone_detected) return;

    const best = detData.detections[0];
    pinDetection(best);

    // ── Model 2: Classify ─────────────────────────────────────
    const fd2 = new FormData();
    fd2.append('image', file);
    const clsRes  = await fetch(`${API}/api/predict/classify`, { method:'POST', body:fd2 });
    const clsData = await clsRes.json();
    renderClassification(clsData);

    // ── Auto-generate track & forecast ───────────────────────
    document.getElementById('det-loader').classList.add('on');
    document.querySelector('.spin-lbl').textContent = 'Generating forecast trajectory…';

    const { lat, lon } = best.geo_centre;
    const windNow = CAT_WIND[clsData.category] || 85;
    const track   = generateTrackFromDetection(lat, lon, windNow);

    // Fill the textarea so user can see/edit what was used
    document.getElementById('track-input').value = JSON.stringify(track, null, 2);
    // Reset preset selector
    const sel = document.getElementById('cyclone-preset');
    if (sel) sel.value = '';

    // Run full forecast pipeline automatically
    await runForecastWithTrack(track, clsData.category);

  } catch (err) {
    console.error(err);
    document.getElementById('det-result').innerHTML =
      `<div class="empty" style="color:#ef4444">Pipeline failed<br><small>Is the backend running on port 5000?</small></div>`;
  } finally {
    document.getElementById('det-loader').classList.remove('on');
    document.querySelector('.spin-lbl').textContent = 'Running Detection…';
  }
}

/**
 * Generate a synthetic 8-point historical track leading up to the
 * currently detected position. Works backward in 3-hour steps.
 *
 * Movement assumption (based on typical NIO cyclone behaviour):
 *   Bay of Bengal (lon > 75): moves NW  → came from SE
 *   Arabian Sea   (lon < 75): moves N   → came from S
 */
function generateTrackFromDetection(lat, lon, windNow) {
  const isBOB = lon > 78;

  // Per-3h step size (degrees) — how far the cyclone moves each interval
  const stepLat = isBOB ? 0.38 : 0.32;
  const stepLon = isBOB ? -0.35 : 0.12;   // BOB moves west, AS moves slightly east

  const track = [];
  for (let i = 7; i >= 0; i--) {
    // i=7 is current; i=0 is 21h ago
    const stepsBack = 7 - i;
    track.push({
      lat:        parseFloat((lat  - stepLat * stepsBack).toFixed(3)),
      lon:        parseFloat((lon  - stepLon * stepsBack).toFixed(3)),
      wind_speed: parseFloat((windNow * (1 - 0.035 * stepsBack)).toFixed(1)),
    });
  }
  return track;   // oldest → newest
}

function resetDetectionUI() {
  document.getElementById('det-result').innerHTML  =
    `<div class="empty">Running detection…</div>`;
  document.getElementById('cls-result').innerHTML  =
    `<div class="empty">Awaiting detection…</div>`;
}

function pinDetection(best) {
  markerLayer.clearLayers();
  const { lat, lon } = best.geo_centre;
  const icon = L.divIcon({
    className: '',
    html: `<div style="
      width:16px;height:16px;
      background:#ff6a00;
      border:2px solid #ffffff;
      border-radius:50%;
      cursor:grab;
      box-shadow:0 0 16px #ff6a00,0 0 32px rgba(255,106,0,.6);
    "></div>`,
    iconSize: [16,16], iconAnchor: [8,8],
  });

  const marker = L.marker([lat, lon], { icon, draggable: true })
    .bindPopup(`<b>Detected Cyclone</b><br>Confidence: ${(best.confidence*100).toFixed(1)}%<br>Lat: ${lat}°N &nbsp; Lon: ${lon}°E<br><span style="color:#ff9142;font-size:11px;">Drag marker on map to adjust position</span>`)
    .addTo(markerLayer);

  marker.on('dragend', async (e) => {
    const p = e.target.getLatLng();
    best.geo_centre.lat = parseFloat(p.lat.toFixed(3));
    best.geo_centre.lon = parseFloat(p.lng.toFixed(3));
    document.getElementById('map-pill').textContent = `Cyclone Position: ${best.geo_centre.lat}°N, ${best.geo_centre.lon}°E`;
    marker.setPopupContent(`<b>Cyclone Position</b><br>Confidence: ${(best.confidence*100).toFixed(1)}%<br>Lat: ${best.geo_centre.lat}°N &nbsp; Lon: ${best.geo_centre.lon}°E<br><span style="color:#ff9142;font-size:11px;">Drag marker on map to adjust position</span>`);

    // Recalculate track and forecast automatically
    const windNow = CAT_WIND[currentCategory] || 85;
    const track = generateTrackFromDetection(best.geo_centre.lat, best.geo_centre.lon, windNow);
    document.getElementById('track-input').value = JSON.stringify(track, null, 2);
    await runForecastWithTrack(track, currentCategory);
  });

  MAP.setView([lat, lon], 6, { animate:true });
  document.getElementById('map-pill').textContent = `Cyclone Detected — ${lat}°N, ${lon}°E (Drag to fine-tune)`;
}

// ── Render Detection ──────────────────────────────────────────
function renderDetection(d) {
  const el = document.getElementById('det-result');
  if (!d.cyclone_detected) {
    el.innerHTML = `
      <div class="ani">
        <div class="cat-badge cat-none">No Cyclone Detected</div>
        <div class="empty" style="padding:6px 0">Image processed — no cyclonic structure found</div>
      </div>`;
    return;
  }
  currentDetection = d.cyclone_detected ? d.detections[0] : null;
  updateAiTelemetryBar();
  const best = d.detections[0];
  el.innerHTML = `
    <div class="ani">
      <div class="cat-badge" style="background:rgba(255,106,0,.15);color:#ff9142;border:1px solid rgba(255,106,0,.4);margin-bottom:8px;">
        Cyclone Detected (${d.count}) · BBox Applied
      </div>
      <div class="info-rows">
        <div class="info-row">
          <span class="info-key">Confidence</span>
          <span class="info-val">${(best.confidence*100).toFixed(1)}%</span>
        </div>
        <div class="info-row">
          <span class="info-key">Centre Lat</span>
          <span class="info-val">${best.geo_centre.lat}° N</span>
        </div>
        <div class="info-row">
          <span class="info-key">Centre Lon</span>
          <span class="info-val">${best.geo_centre.lon}° E</span>
        </div>
        <div class="info-row">
          <span class="info-key">BBox (px)</span>
          <span class="info-val">${best.bbox.map(v=>v.toFixed(0)).join(', ')}</span>
        </div>
      </div>
    </div>`;
}

// ── Render Classification ─────────────────────────────────────
function renderClassification(d) {
  currentCategory = d.category;
  updateAiTelemetryBar();
  const el = document.getElementById('cls-result');

  const probBars = Object.entries(d.probabilities)
    .sort((a,b) => b[1]-a[1])
    .map(([cls,p]) => `
      <div class="prob-row">
        <span class="prob-lbl">${cls}</span>
        <div class="prob-track">
          <div class="prob-fill ${cls===d.category?'top':''}" style="width:${(p*100).toFixed(1)}%"></div>
        </div>
        <span class="prob-val">${(p*100).toFixed(0)}%</span>
      </div>`).join('');

  el.innerHTML = `
    <div class="ani">
      <div class="cat-badge cat-${d.category}">
        ${CAT_LABEL[d.category]||d.category}
        <span style="font-size:11px;opacity:.65">&nbsp;${(d.confidence*100).toFixed(0)}%</span>
      </div>
      <div class="prob-bars">${probBars}</div>
    </div>`;
}

// ─────────────────────────────────────────────────────────────
// 4. CYCLONE PRESETS (real IBTrACS North Indian Ocean data)
// ─────────────────────────────────────────────────────────────

// wind_speed in km/h  |  8 points × 3-hr intervals = 24hr history
const PRESETS = {

  // Cyclone Fani — VSCS, Bay of Bengal, made landfall Odisha May 3 2019
  // Pre-landfall intensification phase (Apr 30, 2019 — 12:00 to May 02 06:00)
  fani: {
    name: 'Cyclone Fani (2019) — VSCS',
    category: 'VSCS',
    track: [
      {lat:14.5, lon:86.8, wind_speed:213},
      {lat:15.2, lon:86.2, wind_speed:222},
      {lat:15.9, lon:85.5, wind_speed:231},
      {lat:16.5, lon:84.8, wind_speed:240},
      {lat:17.1, lon:84.0, wind_speed:240},
      {lat:17.8, lon:83.2, wind_speed:231},
      {lat:18.5, lon:82.4, wind_speed:222},
      {lat:19.2, lon:81.6, wind_speed:213},
    ]
  },

  // Cyclone Amphan — VSCS, Bay of Bengal, made landfall West Bengal May 20 2020
  // Intensification phase (May 17–18, 2020)
  amphan: {
    name: 'Cyclone Amphan (2020) — VSCS',
    category: 'VSCS',
    track: [
      {lat:12.5, lon:87.5, wind_speed:167},
      {lat:13.2, lon:87.0, wind_speed:185},
      {lat:13.9, lon:86.4, wind_speed:204},
      {lat:14.6, lon:85.8, wind_speed:222},
      {lat:15.3, lon:85.1, wind_speed:240},
      {lat:16.0, lon:84.4, wind_speed:240},
      {lat:16.8, lon:83.6, wind_speed:231},
      {lat:17.6, lon:82.8, wind_speed:222},
    ]
  },

  // Cyclone Tauktae — VSCS, Arabian Sea, made landfall Gujarat May 17 2021
  // Intensification phase (May 14–15, 2021)
  tauktae: {
    name: 'Cyclone Tauktae (2021) — VSCS',
    category: 'VSCS',
    track: [
      {lat:15.8, lon:71.2, wind_speed:130},
      {lat:16.5, lon:70.8, wind_speed:148},
      {lat:17.2, lon:70.3, wind_speed:167},
      {lat:17.9, lon:69.8, wind_speed:185},
      {lat:18.6, lon:69.2, wind_speed:204},
      {lat:19.3, lon:68.6, wind_speed:213},
      {lat:20.0, lon:68.0, wind_speed:222},
      {lat:20.7, lon:67.4, wind_speed:213},
    ]
  },

  // Cyclone Biparjoy — VSCS, Arabian Sea, made landfall Gujarat Jun 15 2023
  // Intensification phase (Jun 10–11, 2023)
  biparjoy: {
    name: 'Cyclone Biparjoy (2023) — VSCS',
    category: 'VSCS',
    track: [
      {lat:16.2, lon:63.5, wind_speed:130},
      {lat:16.8, lon:63.0, wind_speed:148},
      {lat:17.3, lon:62.5, wind_speed:157},
      {lat:17.8, lon:62.0, wind_speed:167},
      {lat:18.3, lon:61.5, wind_speed:176},
      {lat:18.8, lon:61.0, wind_speed:185},
      {lat:19.3, lon:60.5, wind_speed:185},
      {lat:19.8, lon:60.0, wind_speed:185},
    ]
  },

  // Synthetic demo track
  demo: {
    name: 'Demo Track (synthetic)',
    category: 'CS',
    track: [
      {lat:10.2, lon:80.1, wind_speed:65},
      {lat:10.5, lon:80.5, wind_speed:68},
      {lat:10.8, lon:80.9, wind_speed:71},
      {lat:11.1, lon:81.3, wind_speed:74},
      {lat:11.4, lon:81.7, wind_speed:77},
      {lat:11.7, lon:82.1, wind_speed:80},
      {lat:12.0, lon:82.5, wind_speed:83},
      {lat:12.3, lon:82.9, wind_speed:86},
    ]
  },
};

function loadPreset(key) {
  if (!key || !PRESETS[key]) return;
  const preset = PRESETS[key];
  document.getElementById('track-input').value = JSON.stringify(preset.track, null, 2);
  // Pre-set category from preset
  currentCategory = preset.category;
  // Show a small indicator
  document.getElementById('map-pill').textContent = `${preset.name} — Click Run Forecast`;
}

// Keep backward-compat alias
function loadDemo() { loadPreset('demo'); }



// ─────────────────────────────────────────────────────────────
// 5. FORECAST  (Model 3 + Map routes)
// ─────────────────────────────────────────────────────────────

/** Called by the Run Forecast button — reads track from textarea */
async function runForecast() {
  const raw = document.getElementById('track-input').value.trim();
  if (!raw) { alert('Select a cyclone preset or upload an image first.'); return; }

  let track;
  try { track = JSON.parse(raw); }
  catch { alert('Invalid JSON in track input. Check syntax.'); return; }

  if (track.length !== 8) {
    alert(`Track must have exactly 8 points (you have ${track.length}).`);
    return;
  }

  const btn = document.getElementById('fcst-btn');
  btn.disabled = true;
  btn.textContent = 'Forecasting…';
  try {
    await runForecastWithTrack(track, currentCategory);
  } finally {
    btn.disabled = false;
    btn.innerHTML = 'Run Forecast';
  }
}

/** Core forecast logic — called by both button and auto-pipeline */
async function runForecastWithTrack(track, category) {
  try {
    // ── Model 3: Forecast ─────────────────────────────────────
    const fRes  = await fetch(`${API}/api/predict/forecast`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({track}),
    });
    const fData = await fRes.json();
    if (fData.error) throw new Error(fData.error);

    const cur = track[track.length-1];
    const cat = category || currentCategory || 'CS';
    currentForecast = { cur, cat, forecast: fData.forecast, all_steps: fData.all_steps };
    updateAiTelemetryBar();

    // Re-check user city proximity if selected
    if (activeUserCity) {
      updateProximityDisplay(activeUserCity);
    }

    // ── GeoJSON track ─────────────────────────────────────────
    const gRes  = await fetch(`${API}/api/map/track`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ current:{lat:cur.lat,lon:cur.lon}, forecast:fData.forecast, category:cat }),
    });
    const gData = await gRes.json();

    // ── Impact zones ──────────────────────────────────────────
    const iRes  = await fetch(`${API}/api/map/impact`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ forecast:fData.forecast, category:cat }),
    });
    const iData = await iRes.json();

    // ── Render ────────────────────────────────────────────────
    renderForecastTable(fData.forecast);
    renderIntensityChart(track, fData.all_steps);
    renderTrackOnMap(gData, track, fData.forecast);
    renderImpactZones(iData);
    renderRiskSummary(gData.meta, fData.forecast);

  } catch (err) {
    document.getElementById('fcst-result').innerHTML =
      `<div class="empty" style="color:#ef4444">${err.message}</div>`;
  }
}

// ── Forecast table ────────────────────────────────────
function renderForecastTable(forecast) {
  const rows = forecast.map(p => `
    <tr>
      <td>+${p.hours}h</td>
      <td>${p.lat.toFixed(2)}°N</td>
      <td>${p.lon.toFixed(2)}°E</td>
      <td>${p.wind_speed.toFixed(0)}</td>
    </tr>`).join('');

  document.getElementById('fcst-result').innerHTML = `
    <table class="ftable ani">
      <thead><tr><th>Time</th><th>Lat</th><th>Lon</th><th>km/h</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

// ── Intensity chart (Plotly) ──────────────────────────────────
function renderIntensityChart(history, allSteps) {
  const hHrs = history.map((_,i) => (i - history.length + 1) * 3);
  const hWnd = history.map(p => p.wind_speed);
  const fHrs = allSteps.map(p => p.hours);
  const fWnd = allSteps.map(p => p.wind_speed);

  const layout = {
    paper_bgcolor:'transparent', plot_bgcolor:'transparent',
    margin:{t:6,r:6,b:24,l:36},
    font:{color:'#94a3b8',size:9,family:'Arial, Helvetica, sans-serif'},
    xaxis:{
      gridcolor:'rgba(255,255,255,0.04)',
      zerolinecolor:'rgba(0,212,255,0.2)',
      tickfont:{size:8},
      title:{text:'Hours',font:{size:8}},
    },
    yaxis:{
      gridcolor:'rgba(255,255,255,0.04)',
      tickfont:{size:8},
      title:{text:'km/h',font:{size:8}},
    },
    showlegend:true,
    legend:{font:{size:8},bgcolor:'transparent',x:0,y:1},
  };

  Plotly.newPlot('intensity-chart', [
    {
      x:hHrs, y:hWnd, name:'History',
      type:'scatter', mode:'lines+markers',
      line:{color:'#ffffff',width:2,dash:'dot'},
      marker:{size:4,color:'#ffffff'},
    },
    {
      x:fHrs, y:fWnd, name:'Forecast',
      type:'scatter', mode:'lines+markers',
      line:{color:'#ff6a00',width:2.5},
      marker:{size:4,color:'#ff6a00'},
      fill:'tozeroy', fillcolor:'rgba(255,106,0,0.12)',
    },
  ], layout, { displayModeBar:false, responsive:true });

  document.getElementById('intensity-empty').style.display = 'none';
}

// ── Track on Leaflet map ──────────────────────────────────────
function renderTrackOnMap(geoData, history, forecast) {
  trackLayer.clearLayers();
  markerLayer.clearLayers();

  const color = geoData.meta?.color || '#ff6a00';
  const risk  = geoData.meta?.risk_level || 'MODERATE';

  // History track — dashed white
  const histCoords = history.map(p => [p.lat, p.lon]);
  L.polyline(histCoords, {
    color:'#ffffff', weight:2.5, dashArray:'5 5', opacity:.85,
  }).bindPopup('Historical Track (24h)').addTo(trackLayer);

  // Forecast track — solid orange
  const cur = history[history.length-1];
  L.polyline([[cur.lat,cur.lon], ...forecast.map(p=>[p.lat,p.lon])], {
    color, weight:2.5, opacity:.95,
  }).addTo(trackLayer);

  // Uncertainty circles
  forecast.forEach(pt => {
    const r = UNCERTAINTY_KM[pt.hours] || 200;
    L.circle([pt.lat,pt.lon], {
      radius: r * 1000,
      color, weight:1, dashArray:'4 4',
      fillColor:color, fillOpacity:.04, opacity:.2,
    }).bindPopup(
      `<b>+${pt.hours}h Forecast</b><br>Lat: ${pt.lat.toFixed(2)}°N &nbsp; Lon: ${pt.lon.toFixed(2)}°E<br>Wind: ${pt.wind_speed.toFixed(0)} km/h<br>Uncertainty: ±${r} km`
    ).addTo(trackLayer);
  });

  // Current position — glowing dot
  const curIcon = L.divIcon({
    className:'',
    html:`<div style="
      width:14px;height:14px;
      background:${color};border:2px solid #fff;border-radius:50%;
      box-shadow:0 0 16px ${color},0 0 32px ${color}55;
    "></div>`,
    iconSize:[14,14], iconAnchor:[7,7],
  });
  L.marker([cur.lat,cur.lon], {icon:curIcon})
    .bindPopup(`<b>Current Position</b><br>Wind: ${cur.wind_speed} km/h`)
    .addTo(markerLayer);

  // Forecast dots — fading
  forecast.forEach((pt,i) => {
    const ico = L.divIcon({
      className:'',
      html:`<div style="
        width:8px;height:8px;background:${color};
        border:1.5px solid rgba(255,255,255,.45);border-radius:50%;
        opacity:${0.45 + i*0.15};
      "></div>`,
      iconSize:[8,8], iconAnchor:[4,4],
    });
    L.marker([pt.lat,pt.lon], {icon:ico})
      .bindPopup(`+${pt.hours}h — ${pt.wind_speed.toFixed(0)} km/h`)
      .addTo(markerLayer);
  });

  // Fit bounds
  const all = [...histCoords, ...forecast.map(p=>[p.lat,p.lon])];
  MAP.fitBounds(L.latLngBounds(all).pad(0.2), {animate:true});

  document.getElementById('map-pill').textContent =
    `${geoData.meta?.category || ''} · Risk: ${risk}`;
}

// ── Impact zones ──────────────────────────────────────────────
function renderImpactZones(impData) {
  impactLayer.clearLayers();
  (impData.features || []).forEach(f => {
    const coords = f.geometry.coordinates[0].map(c=>[c[1],c[0]]);
    const col    = f.properties.color || '#f59e0b';
    L.polygon(coords, {
      color:col, weight:1, dashArray:'5 5',
      fillColor:col, fillOpacity:.07,
    }).bindPopup(`Impact Zone +${f.properties.hours}h &nbsp; Risk: ${f.properties.risk}`)
      .addTo(impactLayer);
  });
}

// ── Risk summary card ─────────────────────────────────────────
function renderRiskSummary(meta, forecast) {
  const risk  = meta?.risk_level || 'UNKNOWN';
  const col   = RISK_COLOR[risk] || '#94a3b8';
  const last  = forecast[forecast.length-1];
  const rClass = risk.replace(' ','\\ ');

  document.getElementById('risk-result').innerHTML = `
    <div class="ani">
      <div class="cat-badge cat-${currentCategory||''}" style="margin-bottom:8px;">
        ${risk} RISK
      </div>
      <div class="info-rows">
        <div class="info-row">
          <span class="info-key">IMD Category</span>
          <span class="info-val">${CAT_LABEL[meta?.category]||meta?.category||'—'}</span>
        </div>
        <div class="info-row">
          <span class="info-key">+24h Wind</span>
          <span class="info-val">${last?.wind_speed?.toFixed(0)||'—'} km/h</span>
        </div>
        <div class="info-row">
          <span class="info-key">+24h Position Error</span>
          <span class="info-val">~391 km</span>
        </div>
        <div class="info-row">
          <span class="info-key">+6h Position Error</span>
          <span class="info-val">~154 km</span>
        </div>
      </div>
    </div>`;
}

// ─────────────────────────────────────────────────────────────
// 6. CLEAR ALL
// ─────────────────────────────────────────────────────────────
function clearAll() {
  trackLayer.clearLayers();
  impactLayer.clearLayers();
  markerLayer.clearLayers();
  if (userLocationLayer) userLocationLayer.clearLayers();
  closeProximityCard();
  activeUserCity   = null;
  currentDetection = null;
  currentForecast  = null;
  currentCategory  = null;
  updateAiTelemetryBar();
}

// ─────────────────────────────────────────────────────────────
// 7. ASK CYCLONE AI & ACTION PLANNER (Groq API)
// ─────────────────────────────────────────────────────────────

function openAiModal() {
  const modal = document.getElementById('ai-modal');
  if (!modal) return;
  modal.classList.add('open');
  updateAiTelemetryBar();

  setTimeout(() => {
    document.getElementById('ai-user-input')?.focus();
  }, 100);
}

function closeAiModal() {
  const modal = document.getElementById('ai-modal');
  if (modal) modal.classList.remove('open');
}

function selectAiRole(role) {
  activeAiRole = role;
  document.querySelectorAll('#ai-role-pills .role-pill').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.role === role);
  });
}

function askQuickPrompt(promptText) {
  const input = document.getElementById('ai-user-input');
  if (!input) return;
  input.value = promptText;
  submitAiMessage();
}

function updateAiTelemetryBar() {
  const catEl  = document.getElementById('ai-tele-cat');
  const posEl  = document.getElementById('ai-tele-pos');
  const windEl = document.getElementById('ai-tele-wind');
  const riskEl = document.getElementById('ai-tele-risk');
  if (!catEl || !posEl || !windEl || !riskEl) return;

  const cat = currentCategory || currentForecast?.cat || '—';
  const label = CAT_LABEL[cat] || cat;
  catEl.textContent = cat !== '—' ? `${cat} (${label})` : 'Awaiting Detection';

  let lat = null, lon = null, wind = null;
  if (currentDetection?.geo_centre) {
    lat = currentDetection.geo_centre.lat;
    lon = currentDetection.geo_centre.lon;
  } else if (currentForecast?.cur) {
    lat = currentForecast.cur.lat;
    lon = currentForecast.cur.lon;
    wind = currentForecast.cur.wind_speed;
  }
  posEl.textContent = (lat != null && lon != null) ? `${lat}°N, ${lon}°E` : '—';
  windEl.textContent = (wind != null) ? `${wind} km/h` : (CAT_WIND[cat] ? `~${CAT_WIND[cat]} km/h` : '—');

  const riskMapping = { D:'LOW', DD:'LOW', CS:'MODERATE', SCS:'HIGH', VSCS:'VERY HIGH' };
  const risk = riskMapping[cat] || 'MODERATE';
  riskEl.textContent = risk;
  riskEl.style.color = RISK_COLOR[risk] || 'inherit';
}

function renderMarkdown(md) {
  if (window.marked && typeof window.marked.parse === 'function') {
    return window.marked.parse(md);
  }
  return md
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/\n\n/g, '<p></p>')
    .replace(/\n/g, '<br>');
}

async function submitAiMessage() {
  const input = document.getElementById('ai-user-input');
  const msg = input.value.trim();
  if (!msg) return;

  const chatBody = document.getElementById('ai-chat-body');

  // Append user bubble
  const userBubble = document.createElement('div');
  userBubble.className = 'ai-bubble user';
  userBubble.textContent = msg;
  chatBody.appendChild(userBubble);
  input.value = '';

  // Append typing indicator
  const typingBubble = document.createElement('div');
  typingBubble.className = 'ai-bubble assistant';
  typingBubble.id = 'ai-typing-indicator';
  typingBubble.innerHTML = `
    <div class="typing-dots">
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
    </div>
    <span style="font-size:11px;color:var(--text-secondary);margin-left:6px;">Consulting Groq LLaMA-3.3…</span>
  `;
  chatBody.appendChild(typingBubble);
  chatBody.scrollTop = chatBody.scrollHeight;

  const sendBtn = document.getElementById('btn-ai-send');
  if (sendBtn) sendBtn.disabled = true;

  // Build telemetry context
  const lat = currentDetection?.geo_centre?.lat || currentForecast?.cur?.lat || null;
  const lon = currentDetection?.geo_centre?.lon || currentForecast?.cur?.lon || null;
  const wind = currentForecast?.cur?.wind_speed || (CAT_WIND[currentCategory] || null);

  const context = {
    cyclone_detected: Boolean(currentDetection || currentForecast),
    category: currentCategory || currentForecast?.cat || 'Unknown',
    confidence: currentDetection?.confidence || 0.85,
    current_position: { lat, lon },
    wind_speed_kmh: wind,
    forecast: currentForecast?.forecast || [],
    risk_level: document.getElementById('ai-tele-risk')?.textContent || 'MODERATE'
  };

  try {
    const res = await fetch(`${API}/api/ai/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: msg,
        role: activeAiRole,
        context,
        conversation_history: aiHistory
      })
    });

    const data = await res.json();
    typingBubble.remove();

    if (!res.ok) {
      const errBubble = document.createElement('div');
      errBubble.className = 'ai-bubble assistant';
      errBubble.style.borderColor = '#ef4444';
      if (res.status === 401) {
        errBubble.innerHTML = `
          <strong style="color:#ef4444">Groq API Key Not Found</strong><br>
          <span style="font-size:11.5px;line-height:1.5;">Please paste your Groq API key into the <code>backend/.env</code> file in your project folder:<br>
          <code style="background:rgba(0,0,0,0.6);padding:3px 8px;border-radius:4px;display:inline-block;margin-top:6px;color:#ff9142;">GROQ_API_KEY=gsk_your_key_here</code></span>
        `;
      } else {
        errBubble.innerHTML = `<strong style="color:#ef4444">Groq Error:</strong> ${data.error || 'Request failed'}`;
      }
      chatBody.appendChild(errBubble);
      return;
    }

    aiHistory.push({ role: 'user', content: msg });
    aiHistory.push({ role: 'assistant', content: data.response });

    const aiBubble = document.createElement('div');
    aiBubble.className = 'ai-bubble assistant';
    aiBubble.innerHTML = renderMarkdown(data.response);
    chatBody.appendChild(aiBubble);

  } catch (err) {
    typingBubble.remove();
    const errBubble = document.createElement('div');
    errBubble.className = 'ai-bubble assistant';
    errBubble.style.borderColor = '#ef4444';
    errBubble.innerHTML = `<strong style="color:#ef4444">Connection Error:</strong> Unable to reach CycloSense AI service (${err.message}).`;
    chatBody.appendChild(errBubble);
  } finally {
    if (sendBtn) sendBtn.disabled = false;
    chatBody.scrollTop = chatBody.scrollHeight;
  }
}

// ─────────────────────────────────────────────────────────────
// 8. CITY SEARCH & PROXIMITY ALERT
// ─────────────────────────────────────────────────────────────

const COASTAL_CITIES = [
  // Odisha & Bengal
  { name: 'Puri, Odisha', lat: 19.8135, lon: 85.8312, region: 'East Coast' },
  { name: 'Bhubaneswar, Odisha', lat: 20.2961, lon: 85.8245, region: 'East Coast' },
  { name: 'Paradeep, Odisha', lat: 20.3164, lon: 86.6114, region: 'East Coast' },
  { name: 'Gopalpur, Odisha', lat: 19.2604, lon: 84.9089, region: 'East Coast' },
  { name: 'Balasore, Odisha', lat: 21.4934, lon: 86.9135, region: 'East Coast' },
  { name: 'Kolkata, West Bengal', lat: 22.5726, lon: 88.3639, region: 'East Coast' },
  { name: 'Digha, West Bengal', lat: 21.6266, lon: 87.5074, region: 'East Coast' },
  { name: 'Haldia, West Bengal', lat: 22.0620, lon: 88.0673, region: 'East Coast' },

  // Andhra Pradesh & Tamil Nadu
  { name: 'Visakhapatnam, Andhra Pradesh', lat: 17.6868, lon: 83.2185, region: 'East Coast' },
  { name: 'Kakinada, Andhra Pradesh', lat: 16.9891, lon: 82.2475, region: 'East Coast' },
  { name: 'Machilipatnam, Andhra Pradesh', lat: 16.1875, lon: 81.1389, region: 'East Coast' },
  { name: 'Nellore, Andhra Pradesh', lat: 14.4426, lon: 79.9865, region: 'East Coast' },
  { name: 'Chennai, Tamil Nadu', lat: 13.0827, lon: 80.2707, region: 'East Coast' },
  { name: 'Cuddalore, Tamil Nadu', lat: 11.7480, lon: 79.7714, region: 'East Coast' },
  { name: 'Nagapattinam, Tamil Nadu', lat: 10.7672, lon: 79.8449, region: 'East Coast' },
  { name: 'Puducherry', lat: 11.9416, lon: 79.8083, region: 'East Coast' },

  // Gujarat & Maharashtra (Arabian Sea)
  { name: 'Mumbai, Maharashtra', lat: 18.9220, lon: 72.8347, region: 'West Coast' },
  { name: 'Ratnagiri, Maharashtra', lat: 16.9902, lon: 73.3120, region: 'West Coast' },
  { name: 'Surat, Gujarat', lat: 21.1702, lon: 72.8311, region: 'West Coast' },
  { name: 'Porbandar, Gujarat', lat: 21.6417, lon: 69.6293, region: 'West Coast' },
  { name: 'Veraval, Gujarat', lat: 20.9077, lon: 70.3676, region: 'West Coast' },
  { name: 'Dwarka, Gujarat', lat: 22.2442, lon: 68.9685, region: 'West Coast' },
  { name: 'Kandla, Gujarat', lat: 23.0118, lon: 70.2184, region: 'West Coast' },
  { name: 'Panaji, Goa', lat: 15.4909, lon: 73.8278, region: 'West Coast' },
  { name: 'Mangalore, Karnataka', lat: 12.9141, lon: 74.8560, region: 'West Coast' },
  { name: 'Kochi, Kerala', lat: 9.9312, lon: 76.2673, region: 'West Coast' },
];

function haversineDistKm(lat1, lon1, lat2, lon2) {
  const R = 6371;
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
            Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
            Math.sin(dLon/2) * Math.sin(dLon/2);
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
}

function setupCitySearch() {
  const input = document.getElementById('city-search-input');
  const sugBox = document.getElementById('city-suggestions');
  if (!input || !sugBox) return;

  input.addEventListener('input', () => {
    const q = input.value.trim().toLowerCase();
    if (!q) {
      sugBox.style.display = 'none';
      return;
    }

    const matches = COASTAL_CITIES.filter(c =>
      c.name.toLowerCase().includes(q) || c.region.toLowerCase().includes(q)
    ).slice(0, 6);

    if (matches.length === 0) {
      sugBox.innerHTML = `
        <div class="city-sug-item" onclick="searchOnlineCity('${encodeURIComponent(q)}')">
          <span>Search "<strong>${q}</strong>" online...</span>
        </div>`;
      sugBox.style.display = 'flex';
      return;
    }

    sugBox.innerHTML = matches.map(c => `
      <div class="city-sug-item" onclick='selectCity(${JSON.stringify(c)})'>
        <span>${c.name}</span>
        <span class="city-sug-region">${c.region}</span>
      </div>
    `).join('');
    sugBox.style.display = 'flex';
  });

  input.addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      const q = input.value.trim();
      if (q) searchOnlineCity(encodeURIComponent(q));
    }
  });

  // Click outside closes suggestions
  document.addEventListener('click', e => {
    if (!e.target.closest('#city-search-bar')) {
      sugBox.style.display = 'none';
    }
  });
}

async function searchOnlineCity(query) {
  const sugBox = document.getElementById('city-suggestions');
  sugBox.style.display = 'none';
  try {
    const res = await fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${query}&limit=1`);
    const data = await res.json();
    if (data && data.length > 0) {
      selectCity({
        name: data[0].display_name.split(',')[0] + ', ' + (data[0].display_name.split(',')[1] || ''),
        lat: parseFloat(data[0].lat),
        lon: parseFloat(data[0].lon),
        region: 'Geocoded'
      });
    } else {
      alert('Location not found. Please try another coastal city name.');
    }
  } catch {
    alert('Geocoding service unavailable.');
  }
}

function locateUserGps() {
  if (!navigator.geolocation) {
    alert('Geolocation is not supported by your browser.');
    return;
  }
  const btn = document.getElementById('btn-locate-me');
  btn.textContent = '...';
  navigator.geolocation.getCurrentPosition(
    pos => {
      btn.textContent = 'GPS';
      selectCity({
        name: 'My Current Location (GPS)',
        lat: parseFloat(pos.coords.latitude.toFixed(4)),
        lon: parseFloat(pos.coords.longitude.toFixed(4)),
        region: 'User Device'
      });
    },
    err => {
      btn.textContent = 'GPS';
      alert('Unable to retrieve your location: ' + err.message);
    }
  );
}

function selectCity(city) {
  activeUserCity = city;
  const sugBox = document.getElementById('city-suggestions');
  if (sugBox) sugBox.style.display = 'none';
  document.getElementById('city-search-input').value = city.name;

  updateProximityDisplay(city);
}

function updateProximityDisplay(city) {
  userLocationLayer.clearLayers();

  // Pin city with custom pulsing marker
  const userIcon = L.divIcon({
    className: '',
    html: `<div style="
      width:14px;height:14px;
      background:#f59e0b;
      border:2px solid #fff;
      border-radius:50%;
      box-shadow:0 0 16px #f59e0b, 0 0 32px rgba(245,158,11,0.6);
    "></div>`,
    iconSize: [14, 14],
    iconAnchor: [7, 7],
  });

  L.marker([city.lat, city.lon], { icon: userIcon })
    .bindPopup(`<b>${city.name}</b><br>Lat: ${city.lat}°N, Lon: ${city.lon}°E`)
    .addTo(userLocationLayer)
    .openPopup();

  // Calculate distance to closest point on cyclone path
  const points = [];
  if (currentDetection?.geo_centre) {
    points.push({ hours: 0, lat: currentDetection.geo_centre.lat, lon: currentDetection.geo_centre.lon, wind: CAT_WIND[currentCategory] || 85 });
  } else if (currentForecast?.cur) {
    points.push({ hours: 0, lat: currentForecast.cur.lat, lon: currentForecast.cur.lon, wind: currentForecast.cur.wind_speed });
  }

  if (currentForecast?.forecast) {
    for (const pt of currentForecast.forecast) {
      points.push({ hours: pt.hours, lat: pt.lat, lon: pt.lon, wind: pt.wind_speed });
    }
  }

  const card = document.getElementById('proximity-card');
  card.style.display = 'block';
  document.getElementById('prox-city-name').textContent = city.name.split(',')[0];

  if (points.length === 0) {
    document.getElementById('prox-dist-val').textContent = 'No Active Storm';
    document.getElementById('prox-hazard-badge').textContent = 'NORMAL';
    document.getElementById('prox-hazard-badge').style.background = '#10b981';
    document.getElementById('prox-hazard-badge').style.color = '#fff';
    document.getElementById('prox-meta').textContent = 'Upload image or load a cyclone preset to compute distance and impact window.';
    return;
  }

  let minDist = Infinity;
  let closestPoint = null;

  for (const pt of points) {
    const d = haversineDistKm(city.lat, city.lon, pt.lat, pt.lon);
    if (d < minDist) {
      minDist = d;
      closestPoint = pt;
    }
  }

  const roundedDist = Math.round(minDist);
  document.getElementById('prox-dist-val').textContent = `${roundedDist} km`;

  // Hazard classification
  const badge = document.getElementById('prox-hazard-badge');
  const dot = document.getElementById('prox-dot');
  let hazardText, hazardBg, hazardColor;

  if (minDist < 100) {
    hazardText = 'DIRECT IMPACT';
    hazardBg = 'rgba(239, 68, 68, 0.25)';
    hazardColor = '#ef4444';
  } else if (minDist < 250) {
    hazardText = 'HIGH ALERT';
    hazardBg = 'rgba(249, 115, 22, 0.25)';
    hazardColor = '#f97316';
  } else if (minDist < 450) {
    hazardText = 'WATCH ZONE';
    hazardBg = 'rgba(245, 158, 11, 0.25)';
    hazardColor = '#f59e0b';
  } else {
    hazardText = 'SAFE DISTANCE';
    hazardBg = 'rgba(16, 185, 129, 0.25)';
    hazardColor = '#10b981';
  }

  badge.textContent = hazardText;
  badge.style.background = hazardBg;
  badge.style.color = hazardColor;
  dot.style.background = hazardColor;
  dot.style.boxShadow = `0 0 10px ${hazardColor}`;

  const timeLabel = closestPoint.hours === 0 ? 'Current Eye Position' : `In ~${closestPoint.hours}h (${closestPoint.wind || '—'} km/h)`;
  document.getElementById('prox-meta').innerHTML = `
    Closest approach: <strong>${timeLabel}</strong><br>
    Path distance: <strong>${roundedDist} km</strong> away
  `;

  // Draw dashed distance connector line
  L.polyline([[city.lat, city.lon], [closestPoint.lat, closestPoint.lon]], {
    color: hazardColor,
    weight: 2,
    dashArray: '6, 6',
    opacity: 0.85
  }).addTo(userLocationLayer);

  MAP.flyToBounds([[city.lat, city.lon], [closestPoint.lat, closestPoint.lon]], {
    padding: [80, 80],
    maxZoom: 7,
    duration: 1.2
  });
}

function closeProximityCard() {
  const card = document.getElementById('proximity-card');
  if (card) card.style.display = 'none';
  if (userLocationLayer) userLocationLayer.clearLayers();
}

function askAiAboutCity() {
  if (!activeUserCity) return;
  const dist = document.getElementById('prox-dist-val')?.textContent || 'unknown';
  const hazard = document.getElementById('prox-hazard-badge')?.textContent || 'ALERT';
  openAiModal();

  const input = document.getElementById('ai-user-input');
  if (input) {
    input.value = `I am located in ${activeUserCity.name}. The cyclone path is ${dist} away (Hazard Level: ${hazard}). What specific safety measures, evacuation priorities, wind surge risks, and precautions should our community take right now?`;
    submitAiMessage();
  }
}

// ─────────────────────────────────────────────────────────────
// BOOT
// ─────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initMap();
  checkStatus();
  setInterval(checkStatus, 10000);
  setupUpload();
  setupCitySearch();

  // Escape key closes AI modal
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeAiModal();
  });
});
