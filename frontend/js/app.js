const API_BASE = '/api';
const POLL_INTERVAL_MS = 2000;
let selectedFile = null;
let currentSessionId = null;
let pollTimer = null;
async function apiFetch(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}
function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer');
  if (!container) return;
  const icons = { success: '✅', error: '❌', info: 'ℹ️' };
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${icons[type] || ''}</span><span>${message}</span>`;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}
function initUploadPage() {
  const zone = document.getElementById('uploadZone');
  if (!zone) return;
  const fileInput = document.getElementById('fileInput');
  const btnUpload = document.getElementById('btnUpload');
  const btnProcess = document.getElementById('btnProcess');
  zone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      selectedFile = e.target.files[0];
      document.getElementById('fileName').textContent = selectedFile.name;
      document.getElementById('fileInfo').classList.add('visible');
      btnUpload.disabled = false;
    }
  });
  btnUpload.addEventListener('click', async () => {
    btnUpload.disabled = true;
    btnUpload.innerHTML = 'Uploading...';
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      const result = await apiFetch('/upload', { method: 'POST', body: formData });
      currentSessionId = result.session_id;
      showToast('Uploaded successfully!', 'success');
      btnUpload.innerHTML = 'Uploaded';
      btnProcess.disabled = false;
    } catch (err) {
      showToast('Upload failed', 'error');
      btnUpload.disabled = false;
      btnUpload.innerHTML = 'Upload';
    }
  });
  btnProcess.addEventListener('click', async () => {
    btnProcess.disabled = true;
    document.getElementById('progressContainer').style.display = 'block';
    showToast('Processing started...', 'info');
    try {
      await apiFetch('/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: currentSessionId }),
      });
      pollTimer = setInterval(async () => {
        const status = await apiFetch(`/status?session_id=${currentSessionId}`);
        if (status.status === 'completed') {
          clearInterval(pollTimer);
          window.location.href = `/results.html?session_id=${currentSessionId}`;
        } else if (status.status === 'failed') {
          clearInterval(pollTimer);
          showToast('Processing failed', 'error');
        }
      }, POLL_INTERVAL_MS);
    } catch (err) {
      showToast('Process failed', 'error');
    }
  });
}
async function loadResults(sessionId) {
  try {
    const data = await apiFetch(`/results?session_id=${sessionId}`);
    document.getElementById('loadingCard').style.display = 'none';
    if (!data.success) {
      setTimeout(() => loadResults(sessionId), 3000);
      return;
    }
    document.getElementById('resultsContainer').style.display = 'block';
    renderReadinessScore(data.readiness_report);
    renderDatasetProfile(data.profile);
    renderCleaningActions(data.cleaning_actions);
    renderEDAPlots(data.eda_results);
    renderDownloads(sessionId);
  } catch (err) {
    document.getElementById('loadingCard').innerHTML = '<p>Error loading results.</p>';
  }
}
function renderReadinessScore(report) {
  if (!report) return;
  const score = report.readiness_score || 0;
  document.getElementById('ringScore').textContent = score;
  const badgeEl = document.getElementById('readinessBadge');
  badgeEl.innerHTML = report.ready_for_training ? 
    '<span class="readiness-badge ready">✅ Ready for ML Training</span>' : 
    '<span class="readiness-badge not-ready">⚠️ Needs Improvement</span>';
}
function renderDatasetProfile(profile) {
  const grid = document.getElementById('profileStats');
  if (!profile || !grid) return;
  const stats = [
    { value: profile.row_count || '0', label: 'Rows' },
    { value: profile.column_count || '0', label: 'Columns' },
    { value: profile.total_missing_values || '0', label: 'Missing Values' },
    { value: profile.total_duplicate_rows || '0', label: 'Duplicate Rows' }
  ];
  grid.innerHTML = stats.map(s => `
    <div class="stat-card">
      <div class="stat-value">${s.value}</div>
      <div class="stat-label">${s.label}</div>
    </div>
  `).join('');
}
function renderCleaningActions(actions) {
  const wrap = document.getElementById('cleaningTableWrap');
  if (!wrap || !actions || actions.length === 0) return;
  let html = '<table class="actions-table"><thead><tr><th>Column</th><th>Action</th><th>Status</th></tr></thead><tbody>';
  actions.forEach(a => {
    html += `<tr><td>${a.column_name}</td><td>${a.action_type}</td><td>${a.status === 'success' ? '✅' : '❌'}</td></tr>`;
  });
  wrap.innerHTML = html + '</tbody></table>';
}
function renderEDAPlots(edaResults) {
  const gallery = document.getElementById('plotsGallery');
  if (!gallery || !edaResults || !edaResults.generated_plots) return;
  gallery.innerHTML = edaResults.generated_plots.map(plotPath => {
    const filename = plotPath.replace(/\\/g, '/').split('/').pop();
    return `<div class="plot-card"><img src="/outputs/eda/${filename}" alt="Plot"></div>`;
  }).join('');
}
function renderDownloads(sessionId) {
  const grid = document.getElementById('downloadGrid');
  if (!grid) return;
  const downloads = [
    { icon: '📄', name: 'Cleaned Dataset', url: `${API_BASE}/download/cleaned?session_id=${sessionId}` },
    { icon: '🔧', name: 'Processed Dataset', url: `${API_BASE}/download/processed?session_id=${sessionId}` },
    { icon: '📋', name: 'Report', url: `${API_BASE}/download/report?session_id=${sessionId}` }
  ];
  grid.innerHTML = downloads.map(d => `
    <a class="download-card" href="${d.url}" download>
      <span class="dl-icon">${d.icon}</span><div class="dl-name">${d.name}</div>
    </a>
  `).join('');
}
document.addEventListener('DOMContentLoaded', () => initUploadPage());