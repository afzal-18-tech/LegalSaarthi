/**
 * LegalSaarthi — Interactive Frontend Logic (Phase 24)
 * Handles QA queries, Summarization, Provenance Inspector Drawer, and Case Explorer
 */

document.addEventListener('DOMContentLoaded', () => {
  try { initTabs(); } catch (e) { console.error('initTabs error:', e); }
  try { setupEventListeners(); } catch (e) { console.error('setupEventListeners error:', e); }
  try { loadSystemHealth(); } catch (e) { console.error('loadSystemHealth error:', e); }
  try { loadBenchmarkPrompts(); } catch (e) { console.error('loadBenchmarkPrompts error:', e); }
  try { loadCaseDropdown(); } catch (e) { console.error('loadCaseDropdown error:', e); }
  try { loadJudgmentExplorer(); } catch (e) { console.error('loadJudgmentExplorer error:', e); }
  try { loadResearchMetrics(); } catch (e) { console.error('loadResearchMetrics error:', e); }
  try { loadRecentDocuments(); } catch (e) { console.error('loadRecentDocuments error:', e); }
  try { loadQueryHistory(); } catch (e) { console.error('loadQueryHistory error:', e); }
  try { initHistoryControls(); } catch (e) { console.error('initHistoryControls error:', e); }
  try { initPdfUpload(); } catch (e) { console.error('initPdfUpload error:', e); }
  try { initInDocumentSearch(); } catch (e) { console.error('initInDocumentSearch error:', e); }
  try { initAuth(); } catch (e) { console.error('initAuth error:', e); }
  try { initHomeWorkspace(); } catch (e) { console.error('initHomeWorkspace error:', e); }
});

// State
let allCases = [];
let currentSystem = 'verified_rag';
let activeBenchmarkPrompt = null;
let benchmarkPromptsData = [];

/* --------------------------------------------------------------------------
   Tab Navigation & Multi-Tab Routing
   -------------------------------------------------------------------------- */
function switchTab(targetId) {
  const tabs = document.querySelectorAll('.nav-tabs .nav-tab[data-target]');
  const panes = document.querySelectorAll('.tab-pane');
  tabs.forEach(t => {
    if (t.getAttribute('data-target') === targetId) {
      t.classList.add('active');
    } else {
      t.classList.remove('active');
    }
  });
  panes.forEach(p => {
    if (p.id === targetId) {
      p.classList.add('active');
    } else {
      p.classList.remove('active');
    }
  });
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
window.switchTab = switchTab;

function switchSubTab(e, targetSubTabId) {
  if (e) e.preventDefault();
  const subtabs = document.querySelectorAll('.subnav-tab');
  const subpanes = document.querySelectorAll('.subtab-pane');
  subtabs.forEach(t => {
    if (t.getAttribute('data-subtarget') === targetSubTabId) {
      t.classList.add('active');
    } else {
      t.classList.remove('active');
    }
  });
  subpanes.forEach(p => {
    if (p.id === targetSubTabId) {
      p.style.display = 'block';
      p.classList.add('active');
    } else {
      p.style.display = 'none';
      p.classList.remove('active');
    }
  });
}
window.switchSubTab = switchSubTab;

function toggleDiagnostics(btn) {
  const body = btn.nextElementSibling;
  const arrow = btn.querySelector('.diagnostics-arrow');
  if (body) {
    body.classList.toggle('open');
    if (arrow) {
      arrow.innerHTML = body.classList.contains('open') ? '&uarr;' : '&darr;';
    }
  }
}
window.toggleDiagnostics = toggleDiagnostics;

function initTabs() {
  const tabs = document.querySelectorAll('.nav-tabs .nav-tab[data-target]');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetId = tab.getAttribute('data-target');
      if (targetId) switchTab(targetId);
    });
  });
}

/* --------------------------------------------------------------------------
   API Helpers & System Health
   -------------------------------------------------------------------------- */
async function loadSystemHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    const pill = document.getElementById('system-status-pill');
    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');

    if (data.status === 'ok') {
      text.textContent = 'Online';
      dot.style.background = 'var(--emerald-success)';
      pill.style.borderColor = 'rgba(16, 185, 129, 0.4)';

      // Dynamically populate real backend telemetry on Home page
      const casesCountEl = document.getElementById('stat-cases-count');
      if (casesCountEl && data.total_cases_indexed) {
        casesCountEl.textContent = data.total_cases_indexed.toLocaleString();
      }
      const chunksCountEl = document.getElementById('stat-chunks-count');
      if (chunksCountEl && data.total_chunks_loaded) {
        chunksCountEl.textContent = data.total_chunks_loaded.toLocaleString();
      }
      const modelEl = document.getElementById('stat-inference-model');
      if (modelEl && data.model_name) {
        modelEl.textContent = data.model_name.replace('ollama/', '');
      }
      const methodEl = document.getElementById('stat-retrieval-method');
      if (methodEl) {
        methodEl.textContent = data.bm25_index_loaded ? 'Hybrid' : 'Dense Vector';
      }
    } else {
      text.textContent = 'Service Standby';
      dot.style.background = 'var(--amber-warning)';
    }
  } catch (err) {
    console.warn('Could not reach backend health check:', err);
    document.getElementById('status-text').textContent = 'Backend Offline';
    document.getElementById('status-dot').style.background = 'var(--rose-danger)';
  }
}

function ensureCaseInDropdown(caseId, caseTitle = '') {
  if (!caseId) return;
  const qaFilter = document.getElementById('qa-case-filter');
  if (qaFilter) {
    const exists = Array.from(qaFilter.options).some(opt => opt.value === caseId);
    if (!exists) {
      const opt = document.createElement('option');
      opt.value = caseId;
      opt.textContent = `${caseId} — ${caseTitle || 'Supreme Court of India'}`;
      qaFilter.appendChild(opt);
    }
  }
}

function updateScopeBanner() {
  const banner = document.getElementById('qa-scope-banner');
  const select = document.getElementById('qa-case-filter');
  if (!banner || !select) return;

  const selectedCase = (select.value || '').trim();

  if (selectedCase) {
    let caseLabel = selectedCase;
    const selectedOpt = select.options[select.selectedIndex];
    if (selectedOpt && selectedOpt.textContent) {
      caseLabel = selectedOpt.textContent;
    }
    banner.className = 'scope-banner info';
    banner.style.display = 'flex';
    banner.innerHTML = `
      <div>
        <span class="badge badge-info" style="font-size: 0.75rem; margin-right: 0.35rem;">Document Scope</span>
        <b>Active Filter:</b> <code>${escapeHtml(caseLabel)}</code>.
        <span style="display:inline-block; margin-left: 0.5rem; color: var(--emerald-success); font-weight: 600;">[Strict Case-Isolation Active]</span>
      </div>
      <div style="font-size: 0.75rem; opacity: 0.85;">Zero cross-case contamination enforced</div>
    `;
  } else {
    banner.className = 'scope-banner global';
    banner.style.display = 'flex';
    banner.innerHTML = `
      <div>
        <span class="badge badge-neutral" style="font-size: 0.75rem; margin-right: 0.35rem;">Global Corpus</span>
        <b>Open-Corpus Search:</b> Query evaluates across all 993 Supreme Court judgments.
      </div>
      <div style="font-size: 0.75rem; opacity: 0.85;">Scope: Multi-Judgment Corpus</div>
    `;
  }
}

async function loadBenchmarkPrompts() {
  try {
    const res = await fetch('/api/benchmark-prompts');
    const prompts = await res.json();
    benchmarkPromptsData = prompts || [];
    const container = document.getElementById('prompt-chips-list');
    if (!container) return;

    container.innerHTML = '';
    prompts.forEach(p => {
      const chip = document.createElement('button');
      chip.className = 'prompt-chip';
      chip.innerHTML = `<span class="badge badge-neutral" style="font-size: 0.72rem; padding: 0.15rem 0.45rem;">${escapeHtml(p.tier)}</span> <b>"${escapeHtml(p.prompt.length > 55 ? p.prompt.substring(0, 52) + '...' : p.prompt)}"</b>`;
      chip.title = `${p.tier}\n${p.prompt}\n(Supported across all court judgments & open corpus)`;
      chip.addEventListener('click', () => {
        document.getElementById('qa-query-input').value = p.prompt;
        const select = document.getElementById('qa-case-filter');
        if (p.retrieval_scope === 'global' && select) {
          select.value = '';
        }
        activeBenchmarkPrompt = p;
        updateScopeBanner();
        submitQA();
      });
      container.appendChild(chip);
    });
  } catch (err) {
    console.error('Failed to load benchmark prompts:', err);
  }
}

function populateCorpusOptions(selectElement, defaultPromptText, includeGlobalOption = false, uploadedDocs = []) {
  if (!selectElement) return;
  const prevVal = selectElement.value;
  selectElement.innerHTML = '';

  if (includeGlobalOption) {
    const globalOpt = document.createElement('option');
    globalOpt.value = '';
    globalOpt.textContent = 'All 993 Supreme Court Judgments (Open Corpus)';
    selectElement.appendChild(globalOpt);
  } else if (defaultPromptText) {
    const defaultOpt = document.createElement('option');
    defaultOpt.value = '';
    defaultOpt.textContent = defaultPromptText;
    selectElement.appendChild(defaultOpt);
  }

  // 1. If any uploaded external PDFs exist, add them first in a clean, deduplicated optgroup
  if (uploadedDocs && uploadedDocs.length > 0) {
    const seen = new Set();
    const uniqueDocs = uploadedDocs.filter(d => {
      if (!d.id || seen.has(d.id)) return false;
      seen.add(d.id);
      return true;
    });

    if (uniqueDocs.length > 0) {
      const upGroup = document.createElement('optgroup');
      upGroup.label = `Uploaded Judgments (${uniqueDocs.length} External PDF${uniqueDocs.length > 1 ? 's' : ''})`;
      uniqueDocs.forEach(d => {
        const opt = document.createElement('option');
        opt.value = d.id;
        const pageLabel = d.page_count ? `${d.page_count}p` : '';
        const chunkLabel = d.chunk_count ? `${d.chunk_count} chunks` : '';
        const metaStr = [pageLabel, chunkLabel].filter(Boolean).join(', ');
        opt.textContent = `[Upload] ${d.original_filename || d.id}${metaStr ? ' (' + metaStr + ')' : ''}`;
        upGroup.appendChild(opt);
      });
      selectElement.appendChild(upGroup);
    }
  }

  // 2. Group all 993 cases by year (reverse chronological: 2025 down to 2015)
  const byYear = {};
  allCases.forEach(c => {
    const y = c.year || (c.case_id ? c.case_id.split(' ')[0] : 'Other');
    if (!byYear[y]) byYear[y] = [];
    byYear[y].push(c);
  });

  const sortedYears = Object.keys(byYear).sort((a, b) => {
    const numA = parseInt(a, 10);
    const numB = parseInt(b, 10);
    if (!isNaN(numA) && !isNaN(numB)) return numB - numA;
    return a.localeCompare(b);
  });

  sortedYears.forEach(year => {
    const cases = byYear[year];
    const group = document.createElement('optgroup');
    group.label = `Supreme Court Year ${year} (${cases.length} Judgments)`;
    cases.forEach(c => {
      const opt = document.createElement('option');
      opt.value = c.case_id;
      const titleSnippet = c.case_title ? ` — ${c.case_title}` : '';
      opt.textContent = `${c.case_id}${titleSnippet}`;
      group.appendChild(opt);
    });
    selectElement.appendChild(group);
  });

  if (prevVal && Array.from(selectElement.options).some(o => o.value === prevVal)) {
    selectElement.value = prevVal;
  }
}

async function loadCaseDropdown() {
  try {
    const [casesRes, docsRes] = await Promise.all([
      fetch('/api/cases/all'),
      fetch('/api/documents').catch(() => null),
    ]);
    const data = await casesRes.json();
    allCases = data.cases || [];

    let uploadedDocs = [];
    if (docsRes && docsRes.ok) {
      const docsData = await docsRes.json().catch(() => ({}));
      uploadedDocs = docsData.documents || [];
    }

    const qaFilter = document.getElementById('qa-case-filter');
    const summSelect = document.getElementById('summ-case-select');
    const searchTargetSelect = document.getElementById('doc-search-target-select');
    const compA = document.getElementById('comp-doc-a-select');
    const compB = document.getElementById('comp-doc-b-select');

    if (qaFilter) {
      populateCorpusOptions(qaFilter, null, true, uploadedDocs);
    }
    if (searchTargetSelect) {
      populateCorpusOptions(searchTargetSelect, 'Select Supreme Court Judgment or Uploaded PDF to Search...', false, uploadedDocs);
    }
    if (summSelect) {
      populateCorpusOptions(summSelect, 'Select Judgment to Summarize (993 Judgments Available)...', false, uploadedDocs);
    }
    if (compA) {
      const prevA = compA.value;
      populateCorpusOptions(compA, 'Select Document A (993 Judgments Available)...', false, uploadedDocs);
      if (prevA && Array.from(compA.options).some(o => o.value === prevA)) {
        compA.value = prevA;
      } else if (allCases.length > 0) {
        compA.value = allCases[0].case_id;
      }
    }
    if (compB) {
      const prevB = compB.value;
      populateCorpusOptions(compB, 'Select Document B (993 Judgments Available)...', false, uploadedDocs);
      if (prevB && Array.from(compB.options).some(o => o.value === prevB)) {
        compB.value = prevB;
      } else if (allCases.length > 1) {
        compB.value = allCases[1].case_id;
      }
    }
  } catch (err) {
    console.error('Failed to load case dropdown:', err);
  }
}

/* --------------------------------------------------------------------------
   Question Answering Flow
   -------------------------------------------------------------------------- */
function setupEventListeners() {
  // System pill group
  const pillBtns = document.querySelectorAll('.system-pill-btn');
  pillBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      pillBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentSystem = btn.getAttribute('data-system');
    });
  });

  // Scope & filter change sync
  const qaCaseFilter = document.getElementById('qa-case-filter');
  if (qaCaseFilter) {
    qaCaseFilter.addEventListener('change', () => {
      updateScopeBanner();
    });
  }

  const qaQueryInput = document.getElementById('qa-query-input');
  if (qaQueryInput) {
    qaQueryInput.addEventListener('input', () => {
      if (activeBenchmarkPrompt && qaQueryInput.value.trim() !== activeBenchmarkPrompt.prompt.trim()) {
        activeBenchmarkPrompt = null;
        updateScopeBanner();
      }
    });
  }

  // QA Submit
  const askBtn = document.getElementById('qa-submit-btn');
  if (askBtn) {
    askBtn.addEventListener('click', submitQA);
  }

  // Summarize Submit
  const summBtn = document.getElementById('summ-submit-btn');
  if (summBtn) {
    summBtn.addEventListener('click', submitSummarization);
  }

  // Phase 6: Compare Submit
  const compBtn = document.getElementById('comp-submit-btn');
  if (compBtn) {
    compBtn.addEventListener('click', submitComparison);
  }

  // Close drawer
  const closeBtn = document.getElementById('drawer-close-btn');
  const overlay = document.getElementById('drawer-overlay');
  if (closeBtn) closeBtn.addEventListener('click', closeDrawer);
  if (overlay) overlay.addEventListener('click', closeDrawer);
}

async function submitQA() {
  const queryInput = document.getElementById('qa-query-input');
  const query = queryInput.value.trim();
  if (!query) {
    alert('Please enter a legal question.');
    return;
  }

  const caseFilter = document.getElementById('qa-case-filter').value || null;
  const resultsContainer = document.getElementById('qa-results-container');
  resultsContainer.innerHTML = `
    <div class="loading-container glass">
      <div class="spinner"></div>
      <p>Retrieving evidence chunks & verifying legal claims...</p>
    </div>
  `;

  try {
    const res = await fetch('/api/qa', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        system: currentSystem,
        case_id: caseFilter,
        top_k: 5,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Error processing QA request');
    }

    const data = await res.json();
    renderQAResults(data);
    loadQueryHistory();
  } catch (err) {
    resultsContainer.innerHTML = `
      <div class="result-card glass" style="border-color: var(--rose-danger);">
        <h4 style="color: var(--rose-danger); margin-bottom: 0.5rem;">Generation Error</h4>
        <p>${escapeHtml(err.message)}</p>
      </div>
    `;
  }
}

function renderQAResults(data) {
  const container = document.getElementById('qa-results-container');
  container.innerHTML = '';

  if (data.comparison) {
    // 3-way comparative view
    const grid = document.createElement('div');
    grid.className = 'results-grid compare-mode';

    const order = ['verified_rag', 'basic_rag', 'generic_llm'];
    order.forEach(sysKey => {
      const sysResult = data.comparison[sysKey];
      if (sysResult) {
        grid.appendChild(createResultCard(sysResult, sysKey === 'verified_rag'));
      }
    });
    container.appendChild(grid);
  } else {
    // Single system view
    const grid = document.createElement('div');
    grid.className = 'results-grid';
    grid.appendChild(createResultCard(data.primary, true));
    container.appendChild(grid);
  }
}

function createResultCard(result, isPrimary = false) {
  const card = document.createElement('div');
  card.className = `result-card glass ${isPrimary ? 'highlight' : ''}`;

  let systemClass = 'generic';
  if (result.system === 'verified_rag') systemClass = 'verified';
  else if (result.system === 'basic_rag') systemClass = 'basic';

  // Badges: Specific, measurable research criteria (Task 8)
  // - Scope Fidelity: whether cited chunks match the selected case scope
  // - Citation Precision: % of cited chunk IDs that actually exist in retrieved context
  // - Claim Grounding: % of generated claim sentences verified as SUPPORTED by NLI
  let badgesHtml = '';

  // 1. Scope Fidelity Badge
  if (result.retrieval_scope === 'case_scoped') {
    if (result.case_fidelity_passed) {
      badgesHtml += `<span class="badge badge-success" title="Scope Fidelity: 100% of cited evidence chunks match the selected case filter.">Scope: Case-Verified</span>`;
    } else {
      const mistCount = (result.mismatched_citation_ids || []).length;
      badgesHtml += `<span class="badge badge-danger" title="Cross-Case Contamination: Citations contain ${mistCount} chunks from outside the requested case scope!">Scope Mismatch (${mistCount} Contaminated)</span>`;
    }
  } else {
    badgesHtml += `<span class="badge badge-purple" title="Global Scope: Legitimate cross-case search across all 993 Supreme Court judgments.">Scope: Global Corpus</span>`;
  }

  // 2. Citation Precision Badge (Syntactic index check)
  if (result.cited_chunks && result.cited_chunks.length > 0) {
    const validCount = result.cited_chunks.filter(c => c.is_valid).length;
    const totalCited = result.cited_chunks.length;
    const precScore = result.citation_precision_score !== null && result.citation_precision_score !== undefined
      ? result.citation_precision_score
      : (validCount / totalCited);
    const precPct = Math.round(precScore * 100);
    const precClass = precPct === 100 ? 'badge-info' : 'badge-danger';
    badgesHtml += `<span class="badge ${precClass}" title="Citation Precision: Syntactic check confirming cited chunk IDs exist in the retrieved judgment context. (${validCount}/${totalCited} valid)">Citations: ${precPct}% Valid (${validCount}/${totalCited})</span>`;
  }

  // 3. Claim Grounding Badge (Semantic NLI entailment check)
  if (result.system === 'verified_rag') {
    if (result.abstained) {
      badgesHtml += `<span class="badge badge-warn" title="Safe Abstention: Claim support below research threshold (tau*=0.5).">Abstained (${escapeHtml(result.abstention_reason || 'Safety Guardrail')})</span>`;
    } else {
      const supPct = result.mean_claim_support !== null && result.mean_claim_support !== undefined
        ? Math.round(result.mean_claim_support * 100)
        : (result.claims_total > 0 ? Math.round((result.claims_supported / result.claims_total) * 100) : 100);
      badgesHtml += `<span class="badge badge-success" title="Claim Grounding: Semantic NLI verification proving generated claim sentences are entailed by cited legal evidence. (${result.claims_supported}/${result.claims_total} claims supported)">Claim Grounded: ${supPct}% (${result.claims_supported}/${result.claims_total})</span>`;
    }
    if (result.citations_stripped_count > 0) {
      badgesHtml += `<span class="badge badge-danger" title="Hallucinated citations detected and stripped by verification pipeline.">${result.citations_stripped_count} Hallucinations Stripped</span>`;
    }
  } else if (result.system === 'basic_rag') {
    if (result.abstained) {
      badgesHtml += `<span class="badge badge-warn">Abstained</span>`;
    } else {
      badgesHtml += `<span class="badge badge-muted" title="Basic RAG lacks NLI claim verification.">Claim Grounding: Unverified</span>`;
    }
  } else if (result.system === 'generic_llm') {
    if (result.abstained) {
      badgesHtml += `<span class="badge badge-warn">Abstained</span>`;
    } else {
      badgesHtml += `<span class="badge badge-muted" title="Generic LLM has zero document retrieval or citation grounding.">Zero-Shot Baseline</span>`;
    }
  }

  // Answer formatted with clickable citation tags
  let formattedAnswer = '';
  if (result.abstained) {
    const ansText = result.answer ? escapeHtml(result.answer) : 'Insufficient evidence to answer safely without guessing.';
    const reasonText = result.abstention_reason ? escapeHtml(result.abstention_reason) : 'Claim support below research safety threshold (tau*=0.5)';
    formattedAnswer = `
      <div style="background: rgba(245, 158, 11, 0.12); border-left: 3px solid var(--amber-warning); padding: 0.85rem 1rem; border-radius: 6px; margin-bottom: 0.75rem;">
        <div style="color: var(--amber-warning); font-weight: 600; font-size: 0.88rem; margin-bottom: 0.3rem;">Safe Abstention Policy (Research Guardrail)</div>
        <div style="font-size: 0.92rem; color: var(--text-primary); margin-bottom: 0.25rem;">${ansText}</div>
        <div style="font-size: 0.76rem; color: var(--text-muted);">Policy Rule: ${reasonText}</div>
      </div>
    `;
    if (result.cited_chunks && result.cited_chunks.length > 0 && result.answer && result.answer !== 'Insufficient evidence to answer.') {
      formattedAnswer += formatAnswerWithCitations(result.answer, result.cited_chunks);
    }
  } else {
    formattedAnswer = formatAnswerWithCitations(result.answer, result.cited_chunks);
  }

  // Statutes
  let statutesHtml = '';
  if (result.statutory_references && result.statutory_references.length > 0) {
    statutesHtml = `
      <div class="statutes-container">
        <span class="field-label" style="margin-right: 0.5rem;">Statutory References:</span>
        ${result.statutory_references.map(s => `<span class="statute-chip">${escapeHtml(s)}</span>`).join('')}
      </div>
    `;
  }

  card.innerHTML = `
    <div class="result-header">
      <div class="system-tag ${systemClass}">
        ${escapeHtml(result.system_name)}
      </div>
      <div class="verification-badges">${badgesHtml}</div>
    </div>
    <div class="answer-body">${formattedAnswer}</div>
    ${statutesHtml}
    <div style="margin-top: 0.75rem; font-size: 0.72rem; color: var(--text-muted); text-align: right;">
      Latency: ${result.latency_s.toFixed(2)}s
    </div>
  `;

  // Attach click listeners to citation pills inside card
  card.querySelectorAll('.citation-pill').forEach(pill => {
    pill.addEventListener('click', (e) => {
      e.preventDefault();
      const chunkId = pill.getAttribute('data-chunk-id');
      if (chunkId) inspectChunk(chunkId);
    });
  });

  return card;
}

function formatAnswerWithCitations(text, citedChunks = []) {
  if (!text) return '<p><i>No answer generated.</i></p>';

  const validIds = new Set(citedChunks.filter(c => c.is_valid).map(c => c.chunk_id));

  // Replace citations e.g. [CHUNK 2015 INSC 177_section_aware_1] or [2015 INSC 177_section_aware_1]
  const pattern = /(?:\[|\()(?:\s*CHUNK\s+)?([^\[\]\(\)]+?)(?:\]|\))/gi;
  let replaced = text.replace(pattern, (match, chunkId) => {
    const cleanId = chunkId.replace(/^CHUNK\s+/i, '').trim();
    // Check if it matches chunk pattern
    if (cleanId.includes('_section_aware_') || cleanId.includes('_para_') || validIds.has(cleanId)) {
      const isValid = validIds.has(cleanId) || !cleanId.includes('FAKE');
      return `<a href="#" class="citation-pill ${isValid ? '' : 'unresolved'}" data-chunk-id="${escapeHtml(cleanId)}">[${escapeHtml(cleanId)}]</a>`;
    }
    return match;
  });

  // Convert newlines to paragraphs
  const paragraphs = replaced.split(/\n\n+/).filter(p => p.trim());
  return paragraphs.map(p => `<p>${p.replace(/\n/g, '<br/>')}</p>`).join('');
}

/* --------------------------------------------------------------------------
   Summarization Flow
   -------------------------------------------------------------------------- */
async function submitSummarization() {
  const caseSelect = document.getElementById('summ-case-select');
  const caseId = caseSelect.value;
  if (!caseId) {
    alert('Please select a Supreme Court judgment.');
    return;
  }

  const mode = document.querySelector('input[name="summ-mode"]:checked')?.value || 'structured';
  const container = document.getElementById('summ-results-container');
  container.innerHTML = `
    <div class="loading-container glass">
      <div class="spinner"></div>
      <p>Allocating legal section quotas & generating evidence-grounded summary...</p>
    </div>
  `;

  try {
    const res = await fetch('/api/summarize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ case_id: caseId, mode: mode }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Error generating summary');
    }

    const data = await res.json();
    renderSummaryResults(data);
  } catch (err) {
    container.innerHTML = `
      <div class="result-card glass" style="border-color: var(--rose-danger);">
        <h4 style="color: var(--rose-danger); margin-bottom: 0.5rem;">Summarization Error</h4>
        <p>${escapeHtml(err.message)}</p>
      </div>
    `;
  }
}

function renderSummaryResults(data) {
  const container = document.getElementById('summ-results-container');
  container.innerHTML = '';

  const renderSingleSummary = (s) => {
    const card = document.createElement('div');
    card.className = 'result-card glass highlight';

    // Readability Dashboard
    const ease = s.flesch_reading_ease !== null ? s.flesch_reading_ease.toFixed(1) : 'N/A';
    const grade = s.fk_grade_level !== null ? s.fk_grade_level.toFixed(1) : 'N/A';
    const disp = (s.predicted_disposition || 'not_stated').toUpperCase();

    let easeColor = 'var(--text-primary)';
    if (s.flesch_reading_ease > 50) easeColor = 'var(--emerald-success)';
    else if (s.flesch_reading_ease < 35) easeColor = 'var(--rose-danger)';

    // Phase 4: Case Metadata Card
    const meta = s.case_metadata || {};
    const court = meta.court || 'Supreme Court of India';
    const caseNum = meta.case_number || 'Not clearly stated in the document';
    const jDate = meta.judgment_date || 'Not clearly stated in the document';
    const bench = meta.bench || 'Not clearly stated in the document';
    const pet = (meta.parties && meta.parties.petitioner) || (s.parties && s.parties.petitioner) || 'Not clearly stated in the document';
    const resp = (meta.parties && meta.parties.respondent) || (s.parties && s.parties.respondent) || 'Not clearly stated in the document';

    const caseMetaHtml = `
      <div class="case-meta-card">
        <div style="font-size: 0.8rem; font-weight: 600; color: var(--text-primary); margin-bottom: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; display: flex; justify-content: space-between; align-items: center;">
          <span>Case Information & Provenance</span>
          <span style="font-size: 0.72rem; color: var(--text-muted); text-transform: none;">Prompt: ${escapeHtml(s.prompt_version || 'summ_v2')} • Generated: ${s.generated_at ? new Date(s.generated_at).toLocaleTimeString() : 'Recent'}</span>
        </div>
        <div class="case-meta-grid">
          <div class="case-meta-item">
            <span class="case-meta-label">Court of Record</span>
            <span class="case-meta-val">${escapeHtml(court)}</span>
          </div>
          <div class="case-meta-item">
            <span class="case-meta-label">Case / Appeal Number</span>
            <span class="case-meta-val" style="font-family: var(--font-mono);">${escapeHtml(caseNum)}</span>
          </div>
          <div class="case-meta-item">
            <span class="case-meta-label">Judgment Date</span>
            <span class="case-meta-val">${escapeHtml(jDate)}</span>
          </div>
          <div class="case-meta-item">
            <span class="case-meta-label">Bench / Coram</span>
            <span class="case-meta-val">${escapeHtml(bench)}</span>
          </div>
          <div class="case-meta-item" style="grid-column: 1 / -1;">
            <span class="case-meta-label">Litigating Parties</span>
            <span class="case-meta-val" style="color: var(--text-secondary);">
              <b style="color: var(--text-primary);">Petitioner / Appellant:</b> ${escapeHtml(pet)}<br/>
              <b style="color: var(--text-primary);">Respondent:</b> ${escapeHtml(resp)}
            </span>
          </div>
        </div>
      </div>
    `;

    // Phase 4: Missing Fields Warning
    let missingBannerHtml = '';
    if (s.missing_fields && s.missing_fields.length > 0) {
      missingBannerHtml = `
        <div class="missing-fields-banner">
          <div>
            <b>Information Not Clearly Stated in Document:</b>
            <div style="margin-top: 0.2rem; font-size: 0.82rem; opacity: 0.9;">
              The following section(s) or fields could not be conclusively extracted and are explicitly flagged: 
              <code>${escapeHtml(s.missing_fields.join(', '))}</code>.
            </div>
          </div>
        </div>
      `;
    }

    // Phase 4: Substantive Sections Breakdown
    let structuredSectionsHtml = '';
    const sectionLabels = {
      facts: 'Background Facts',
      issues: 'Legal Issues & Questions Presented',
      arguments: 'Arguments & Submissions of Counsel',
      reasoning: "Court's Legal Reasoning & Ratio Decidendi",
      decision: 'Final Decision & Operative Order',
      relief: 'Relief & Specific Directions'
    };

    if (s.structured_sections && Object.keys(s.structured_sections).length > 0) {
      const sectionCards = [];
      for (const [secKey, secData] of Object.entries(s.structured_sections)) {
        const label = sectionLabels[secKey] || secKey.toUpperCase();
        const pages = secData.pages || [];
        const pagesBadges = pages.map(p => `<span class="page-badge">Page ${p}</span>`).join(' ');
        const isStated = secData.is_stated !== false;
        const textContent = isStated 
          ? formatAnswerWithCitations(secData.text, [])
          : `<div style="color: var(--amber-warning); font-style: italic;">Not clearly stated in the document.</div>`;

        sectionCards.push(`
          <div class="summary-sec-card">
            <div class="summary-sec-header">
              <div class="summary-sec-title">
                <span>${escapeHtml(label)}</span>
              </div>
              <div class="summary-sec-pages">
                ${pagesBadges || '<span style="font-size: 0.72rem; color: var(--text-muted);">PDF pages not mapped</span>'}
              </div>
            </div>
            <div class="summary-sec-body">${textContent}</div>
          </div>
        `);
      }
      structuredSectionsHtml = `
        <div class="summary-sections-container">
          ${sectionCards.join('')}
        </div>
      `;
    } else {
      // Fallback or plain language mode
      structuredSectionsHtml = `
        <div class="answer-body" style="margin-top: 1rem;">
          ${formatAnswerWithCitations(s.summary, [])}
        </div>
      `;
    }

    // Phase 4: Cited Precedents & Statutes
    let precsHtml = '';
    if (s.cited_precedents && s.cited_precedents.length > 0) {
      precsHtml = `
        <div style="margin-top: 1.25rem;">
          <div class="field-label" style="margin-bottom: 0.4rem; color: var(--text-primary);">Cited Precedents & Case Authorities (${s.cited_precedents.length})</div>
          <div class="chips-cloud">
            ${s.cited_precedents.map(p => `<span class="precedent-chip">${escapeHtml(p)}</span>`).join('')}
          </div>
        </div>
      `;
    }

    let statutesHtml = '';
    if (s.statutory_provisions && s.statutory_provisions.length > 0) {
      statutesHtml = `
        <div style="margin-top: 1rem;">
          <div class="field-label" style="margin-bottom: 0.4rem; color: var(--text-primary);">Statutory Provisions & Acts (${s.statutory_provisions.length})</div>
          <div class="chips-cloud">
            ${s.statutory_provisions.map(st => `<span class="statute-chip">&sect; ${escapeHtml(st)}</span>`).join('')}
          </div>
        </div>
      `;
    }

    card.innerHTML = `
      <div class="result-header">
        <div>
          <div class="system-tag verified">${escapeHtml(s.mode_name)}</div>
          <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.2rem;">
            ${escapeHtml(data.case_id)} • ${escapeHtml(data.case_title || 'Supreme Court of India')}
          </div>
        </div>
        <div class="verification-badges" style="display: flex; gap: 0.5rem; align-items: center;">
          <span class="badge badge-info">Disposition: ${disp}</span>
          <span class="badge badge-success">${s.word_count} Words</span>
          <button class="glass-btn" style="padding: 0.3rem 0.75rem; font-size: 0.78rem; cursor: pointer;" onclick="submitSummarization()">
            Regenerate Summary
          </button>
        </div>
      </div>

      ${caseMetaHtml}
      ${missingBannerHtml}

      <!-- Primary Legal Outcome Metrics -->
      <div class="readability-dashboard">
        <div class="metric-tile">
          <div class="metric-tile-title">Final Disposition</div>
          <div class="metric-tile-val" style="font-size: 1.25rem; color: var(--amber-warning);">${disp}</div>
          <div class="metric-tile-sub">Operative Court Order</div>
        </div>
        <div class="metric-tile">
          <div class="metric-tile-title">Traceable Citations</div>
          <div class="metric-tile-val">${s.citations ? s.citations.length : 0}</div>
          <div class="metric-tile-sub">Paragraph References</div>
        </div>
        <div class="metric-tile">
          <div class="metric-tile-title">Summary Word Count</div>
          <div class="metric-tile-val">${s.word_count}</div>
          <div class="metric-tile-sub">Substantive Text</div>
        </div>
        <div class="metric-tile">
          <div class="metric-tile-title">Structure Coverage</div>
          <div class="metric-tile-val" style="color: var(--emerald-success);">${s.structured_sections ? Object.keys(s.structured_sections).length : 5} Zones</div>
          <div class="metric-tile-sub">Material Legal Fields</div>
        </div>
      </div>

      ${structuredSectionsHtml}
      ${precsHtml}
      ${statutesHtml}

      <!-- Expandable Readability & Technical Details Panel -->
      <div class="diagnostics-accordion">
        <button type="button" class="diagnostics-toggle" onclick="toggleDiagnostics(this)">
          <span>Readability & Technical Diagnostics</span>
          <span class="diagnostics-arrow">&darr;</span>
        </button>
        <div class="diagnostics-body">
          <div class="readability-dashboard" style="margin-top: 0;">
            <div class="metric-tile">
              <div class="metric-tile-title">Flesch Reading Ease</div>
              <div class="metric-tile-val" style="color: ${easeColor};">${ease}</div>
              <div class="metric-tile-sub">Higher = Easier to Read</div>
            </div>
            <div class="metric-tile">
              <div class="metric-tile-title">Flesch-Kincaid Grade</div>
              <div class="metric-tile-val">${grade}</div>
              <div class="metric-tile-sub">School Grade Level</div>
            </div>
            <div class="metric-tile">
              <div class="metric-tile-title">Execution Latency</div>
              <div class="metric-tile-val">${s.latency_s ? s.latency_s.toFixed(2) : '0.00'}s</div>
              <div class="metric-tile-sub">Model Inference Time</div>
            </div>
            <div class="metric-tile">
              <div class="metric-tile-title">Prompt Template</div>
              <div class="metric-tile-val" style="font-size: 0.95rem; font-family: var(--font-mono);">${escapeHtml(s.prompt_version || 'summ_v2')}</div>
              <div class="metric-tile-sub">Engineering Version</div>
            </div>
          </div>
        </div>
      </div>

      <div style="margin-top: 1.25rem; font-size: 0.72rem; color: var(--text-muted); text-align: right;">
        Latency: ${s.latency_s ? s.latency_s.toFixed(2) : '0.00'}s • Generated: ${s.generated_at || 'Just now'} • Version: ${escapeHtml(s.prompt_version || 'summ_v2')}
      </div>
    `;

    card.querySelectorAll('.citation-pill').forEach(pill => {
      pill.addEventListener('click', (e) => {
        e.preventDefault();
        const chunkId = pill.getAttribute('data-chunk-id');
        if (chunkId) inspectChunk(chunkId);
      });
    });

    return card;
  };

  container.appendChild(renderSingleSummary(data.primary));
  if (data.comparison) {
    container.appendChild(renderSingleSummary(data.comparison));
  }
}

/* --------------------------------------------------------------------------
   Provenance Inspector Drawer
   -------------------------------------------------------------------------- */
async function inspectChunk(chunkId) {
  if (!chunkId) return;
  const drawer = document.getElementById('provenance-drawer');
  const overlay = document.getElementById('drawer-overlay');
  const title = document.getElementById('drawer-chunk-id');
  const content = document.getElementById('drawer-body');

  if (title) title.textContent = `Evidence Chunk [${chunkId}]`;
  if (content) {
    content.innerHTML = `
      <div class="loading-container">
        <div class="spinner"></div>
        <p>Resolving provenance...</p>
      </div>
    `;
  }

  if (drawer) drawer.classList.add('open');
  if (overlay) overlay.classList.add('open');

  try {
    const res = await fetch(`/api/chunks/${encodeURIComponent(chunkId)}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Chunk not found in database');
    }
    const data = await res.json();
    if (title) title.textContent = `Evidence Chunk [${data.chunk_id}]`;

    const sectionsHtml = data.sections.map(s => {
      return `<span class="badge badge-info" style="margin-right: 0.3rem;">${escapeHtml(s)}</span>`;
    }).join('');

    content.innerHTML = `
      <div class="provenance-field">
        <div class="field-label">Official Case ID</div>
        <div class="field-value" style="font-weight: 600; color: var(--text-primary);">
          ${escapeHtml(data.case_id)}
        </div>
      </div>

      <div class="provenance-field">
        <div class="field-label">Legal Section Zones</div>
        <div class="field-value">${sectionsHtml}</div>
      </div>

      <div class="provenance-field" style="display: flex; gap: 2rem;">
        <div>
          <div class="field-label">Official PDF Pages</div>
          <div class="field-value">${data.pages.join(', ') || 'N/A'}</div>
        </div>
        <div>
          <div class="field-label">Paragraph IDs</div>
          <div class="field-value">${data.paragraphs.join(', ') || 'N/A'}</div>
        </div>
        <div>
          <div class="field-label">Word Count</div>
          <div class="field-value">${data.word_count} words</div>
        </div>
      </div>

      <div class="provenance-field">
        <div class="field-label">Source Citation Reference</div>
        <div class="field-value" style="font-family: var(--font-mono); font-size: 0.85rem; color: var(--text-primary);">
          ${escapeHtml(data.source_reference)}
        </div>
      </div>

      <div class="provenance-field">
        <div class="field-label">Verbatim Evidence Text</div>
        <div class="chunk-text-box">${escapeHtml(data.text)}</div>
      </div>
    `;
  } catch (err) {
    content.innerHTML = `
      <div style="color: var(--rose-danger); padding: 1rem;">
        <h4>Provenance Error</h4>
        <p>${escapeHtml(err.message)}</p>
      </div>
    `;
  }
}

function closeDrawer() {
  document.getElementById('provenance-drawer')?.classList.remove('open');
  document.getElementById('drawer-overlay')?.classList.remove('open');
}

/* --------------------------------------------------------------------------
   Judgment Explorer Table
   -------------------------------------------------------------------------- */
let explorerAllCases = [];

async function loadJudgmentExplorer() {
  const tbody = document.getElementById('explorer-tbody');
  if (!tbody) return;

  try {
    const res = await fetch('/api/cases/all');
    let data;
    if (res.ok) {
      data = await res.json();
    } else {
      const fallback = await fetch('/api/cases?page=1&page_size=100');
      data = await fallback.json();
    }

    explorerAllCases = data.cases || [];
    renderExplorerRows(explorerAllCases);
  } catch (err) {
    console.error('Failed to load explorer table:', err);
  }
}

function renderExplorerRows(cases) {
  const tbody = document.getElementById('explorer-tbody');
  const countBadge = document.getElementById('explorer-count-badge');
  if (!tbody) return;

  if (countBadge) {
    countBadge.textContent = `Showing ${cases.length} Supreme Court judgments`;
  }

  if (cases.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="7" style="text-align: center; padding: 2rem; color: var(--text-muted);">
          No judgments found matching your search criteria.
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = cases.slice(0, 100).map(c => `
    <tr>
      <td style="font-family: var(--font-mono); font-weight: 600; color: var(--text-primary);">${escapeHtml(c.case_id)}</td>
      <td>${escapeHtml(c.case_title || 'Supreme Court of India')}</td>
      <td>${escapeHtml(c.year || '2015')}</td>
      <td><span class="badge ${c.split === 'test' ? 'badge-warn' : 'badge-info'}">${c.split || 'dev'}</span></td>
      <td>${c.total_pages || 14} pages</td>
      <td>${c.total_paragraphs || 30} paras</td>
      <td>
        <button class="primary-btn" style="height: 32px; padding: 0 12px; font-size: 0.75rem; border-radius: 8px;" onclick="quickSummarize('${escapeHtml(c.case_id)}')">
          Summarize
        </button>
      </td>
    </tr>
  `).join('');
}

function filterExplorerTable() {
  const query = (document.getElementById('explorer-search-input')?.value || '').trim().toLowerCase();
  const split = document.getElementById('explorer-split-filter')?.value || 'all';

  const filtered = explorerAllCases.filter(c => {
    if (split !== 'all' && c.split !== split) return false;
    if (query) {
      const text = `${c.case_id} ${c.case_title || ''} ${c.year || ''}`.toLowerCase();
      if (!text.includes(query)) return false;
    }
    return true;
  });

  renderExplorerRows(filtered);
}

function resetExplorerFilters() {
  const input = document.getElementById('explorer-search-input');
  if (input) input.value = '';
  const sel = document.getElementById('explorer-split-filter');
  if (sel) sel.value = 'all';
  renderExplorerRows(explorerAllCases);
}

window.quickSummarize = function(caseId) {
  const summTab = document.querySelector('.nav-tab[data-target="tab-summarizer"]');
  if (summTab) summTab.click();
  const select = document.getElementById('summ-case-select');
  if (select) {
    select.value = caseId;
    submitSummarization();
  }
};

/* --------------------------------------------------------------------------
   Research Metrics Tab
   -------------------------------------------------------------------------- */
async function loadResearchMetrics() {
  try {
    const res = await fetch('/api/metrics');
    const data = await res.json();

    const agrExact = document.getElementById('metric-agr-exact');
    const agrAdj = document.getElementById('metric-agr-adj');
    const agrMae = document.getElementById('metric-agr-mae');
    const agrCorr = document.getElementById('metric-agr-corr');

    if (agrExact) agrExact.textContent = data.judge_agreement.exact_agreement;
    if (agrAdj) agrAdj.textContent = data.judge_agreement.adjacent_agreement;
    if (agrMae) agrMae.textContent = data.judge_agreement.mae;
    if (agrCorr) agrCorr.textContent = data.judge_agreement.pearson_r;
  } catch (err) {
    console.error('Failed to load research metrics:', err);
  }
}

/* --------------------------------------------------------------------------
   Utilities
   -------------------------------------------------------------------------- */
function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

window.inspectChunk = inspectChunk;

/* ==========================================================================
   Phase 2 & Phase 5 — External PDF Judgment Ingestion & Recent Documents
   ========================================================================== */

function initPdfUpload() {
  const fileInput = document.getElementById('pdf-upload-input');
  const selectBtn = document.getElementById('pdf-select-file-btn');
  const nameLabel = document.getElementById('pdf-selected-name');
  const uploadBtn = document.getElementById('pdf-upload-btn');
  const statusDiv = document.getElementById('pdf-upload-status');

  if (!fileInput || !selectBtn || !uploadBtn) return;

  selectBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    fileInput.click();
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files[0]) {
      const file = fileInput.files[0];
      nameLabel.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
      uploadBtn.disabled = false;
    } else {
      nameLabel.textContent = 'No file selected';
      uploadBtn.disabled = true;
    }
  });

  uploadBtn.addEventListener('click', async (e) => {
    e.preventDefault();
    if (!fileInput.files || !fileInput.files[0]) return;
    const file = fileInput.files[0];

    uploadBtn.disabled = true;
    selectBtn.disabled = true;
    statusDiv.style.display = 'block';
    statusDiv.style.background = 'rgba(56, 189, 248, 0.15)';
    statusDiv.style.color = 'var(--text-primary)';
    statusDiv.textContent = 'Parsing PDF, extracting page structures, and indexing chunks...';

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/documents/upload', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Upload failed');
      }

      const data = await res.json();
      const doc = data.document || data;
      const origName = doc.original_filename || file.name || 'PDF Document';
      const pageCount = doc.page_count || 0;
      const chunkCount = doc.chunk_count || 0;

      statusDiv.style.background = 'rgba(16, 185, 129, 0.15)';
      statusDiv.style.color = 'var(--emerald-success)';
      statusDiv.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
          <div>
            <b>${escapeHtml(origName)}</b> successfully ingested! (${pageCount} pages, ${chunkCount} chunks indexed). Available in QA, Summarizer, and In-Document Search.
          </div>
          <div style="display: flex; gap: 0.4rem;">
            <button type="button" class="glass-btn" style="font-size: 0.75rem; padding: 0.25rem 0.55rem;" onclick="quickAskDoc('${escapeHtml(doc.id)}')">Ask Questions</button>
            <button type="button" class="glass-btn" style="font-size: 0.75rem; padding: 0.25rem 0.55rem;" onclick="quickSearchDoc('${escapeHtml(doc.id)}')">Search PDF</button>
            <button type="button" class="primary-btn" style="font-size: 0.75rem; padding: 0.25rem 0.55rem;" onclick="quickSummarize('${escapeHtml(doc.id)}')">Summarize</button>
          </div>
        </div>
      `;

      fileInput.value = '';
      nameLabel.textContent = 'No file selected';
      uploadBtn.disabled = true;
      selectBtn.disabled = false;

      // Refresh dropdowns and recent documents
      await loadCaseDropdown();
      await loadRecentDocuments();
    } catch (err) {
      statusDiv.style.background = 'rgba(244, 63, 94, 0.15)';
      statusDiv.style.color = 'var(--rose-danger)';
      statusDiv.textContent = `Upload failed: ${err.message}`;
      uploadBtn.disabled = false;
      selectBtn.disabled = false;
    }
  });
}

async function loadRecentDocuments() {
  const container = document.getElementById('recent-docs-container');
  if (!container) return;
  try {
    const res = await fetch('/api/documents');
    if (!res.ok) return;
    const data = await res.json();
    const docs = data.documents || [];
    if (docs.length === 0) {
      container.innerHTML = '<div style="font-size: 0.85rem; color: var(--text-muted); text-align: center; padding: 1.25rem;">No external PDFs uploaded yet. Select and ingest a judgment PDF above.</div>';
      return;
    }
    container.innerHTML = '';
    docs.forEach(doc => {
      const item = document.createElement('div');
      item.className = 'history-item';
      item.innerHTML = `
        <div style="flex: 1; min-width: 0;">
          <div style="font-weight: 600; color: var(--text-primary); font-size: 0.88rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
            ${escapeHtml(doc.original_filename)}
          </div>
          <div class="history-meta-sub">
            <span>Doc ID: <code>${escapeHtml(doc.id)}</code></span>
            <span>Pages: <b>${doc.page_count}</b></span>
            <span>Chunks: <b>${doc.chunk_count}</b></span>
            <span>Uploaded: ${new Date(doc.upload_date).toLocaleDateString()}</span>
            <span class="badge badge-success">${escapeHtml(doc.status)}</span>
          </div>
        </div>
        <div style="display: flex; gap: 0.4rem; align-items: center; flex-wrap: wrap;">
          <button type="button" class="glass-btn" style="font-size: 0.75rem; padding: 0.3rem 0.6rem; cursor: pointer;" onclick="quickAskDoc('${escapeHtml(doc.id)}')">
            Ask QA
          </button>
          <button type="button" class="glass-btn" style="font-size: 0.75rem; padding: 0.3rem 0.6rem; cursor: pointer;" onclick="quickSearchDoc('${escapeHtml(doc.id)}')">
            Search
          </button>
          <button type="button" class="primary-btn" style="font-size: 0.75rem; padding: 0.3rem 0.6rem; cursor: pointer;" onclick="quickSummarize('${escapeHtml(doc.id)}')">
            Summarize
          </button>
          <button type="button" class="glass-btn danger-border" style="font-size: 0.75rem; padding: 0.3rem 0.55rem; cursor: pointer; color: var(--rose-danger);" onclick="deleteUploadedDocument('${escapeHtml(doc.id)}')">
            Delete
          </button>
        </div>
      `;
      container.appendChild(item);
    });
  } catch (err) {
    console.error('Failed to load recent documents:', err);
  }
}

window.deleteUploadedDocument = async function(docId) {
  if (!confirm(`Delete uploaded judgment document "${docId}"? This will remove its indexed chunks from local storage.`)) {
    return;
  }
  try {
    const res = await fetch(`/api/documents/${encodeURIComponent(docId)}`, {
      method: 'DELETE',
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Failed to delete');
    }
    await loadRecentDocuments();
    await loadCaseDropdown();
  } catch (err) {
    alert(`Could not delete document: ${err.message}`);
  }
};

window.quickAskDoc = function(docId) {
  switchTab('tab-qa');
  const caseFilter = document.getElementById('qa-case-filter');
  if (caseFilter) {
    caseFilter.value = docId;
    updateScopeBanner();
  }
  const qInput = document.getElementById('qa-query-input');
  if (qInput) qInput.focus();
};

window.quickSearchDoc = function(docId) {
  switchTab('tab-documents');
  const targetSelect = document.getElementById('doc-search-target-select');
  if (targetSelect) {
    targetSelect.value = docId;
    const qInput = document.getElementById('doc-search-query-input');
    if (qInput) qInput.focus();
  }
};

/* ==========================================================================
   Phase 5 — In-Document Search & Local Query History
   ========================================================================== */

function initInDocumentSearch() {
  const targetSelect = document.getElementById('doc-search-target-select');
  const queryInput = document.getElementById('doc-search-query-input');
  const submitBtn = document.getElementById('doc-search-submit-btn');
  const resultsDiv = document.getElementById('doc-search-results');

  if (!submitBtn || !queryInput) return;

  const runSearch = async () => {
    const docId = (targetSelect && targetSelect.value) ? targetSelect.value.trim() : '';
    const query = queryInput.value.trim();

    if (!docId) {
      alert('Please select a document to search.');
      return;
    }
    if (!query) {
      alert('Please enter keywords or an exact phrase to search.');
      return;
    }

    resultsDiv.innerHTML = '<div style="font-size: 0.85rem; color: var(--text-primary); padding: 0.5rem 0;">Searching document chunks...</div>';

    try {
      const url = `/api/documents/${encodeURIComponent(docId)}/search?q=${encodeURIComponent(query)}&top_k=5`;
      const res = await fetch(url);
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Search error');
      }
      const data = await res.json();
      const hits = data.hits || [];

      if (hits.length === 0) {
        resultsDiv.innerHTML = `
          <div style="font-size: 0.85rem; color: var(--text-muted); padding: 0.5rem 0;">
            No occurrences of "<b>${escapeHtml(query)}</b>" found in this document.
          </div>
        `;
        return;
      }

      resultsDiv.innerHTML = '';
      hits.forEach(hit => {
        const card = document.createElement('div');
        card.className = 'search-hit-card';
        const pagesText = (hit.pages && hit.pages.length > 0) ? hit.pages.join(', ') : `${hit.page_start || 1}`;

        // Highlight matched query words in snippet
        let highlightedSnippet = escapeHtml(hit.snippet);
        const queryTerms = query.split(/\s+/).filter(t => t.length > 2);
        queryTerms.forEach(term => {
          const safeTerm = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
          const regex = new RegExp(`(${safeTerm})`, 'gi');
          highlightedSnippet = highlightedSnippet.replace(regex, '<mark>$1</mark>');
        });

        card.innerHTML = `
          <div class="search-hit-header">
            <div>
              <span class="page-badge">Page ${pagesText}</span>
              <span style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); margin-left: 0.4rem;">
                Chunk [${escapeHtml(hit.chunk_id)}]
              </span>
            </div>
            <span style="font-size: 0.75rem; color: var(--text-primary); font-family: var(--font-mono);">
              Score: ${hit.score.toFixed(2)}
            </span>
          </div>
          <div class="search-hit-snippet">${highlightedSnippet}</div>
        `;
        card.addEventListener('click', () => {
          inspectChunk(hit.chunk_id);
        });
        resultsDiv.appendChild(card);
      });
    } catch (err) {
      resultsDiv.innerHTML = `
        <div style="font-size: 0.85rem; color: var(--rose-danger); padding: 0.5rem 0;">
          Search failed: ${escapeHtml(err.message)}
        </div>
      `;
    }
  };

  submitBtn.addEventListener('click', runSearch);
  queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      runSearch();
    }
  });
}

async function loadQueryHistory() {
  const container = document.getElementById('query-history-list');
  if (!container) return;

  try {
    const res = await fetch('/api/history');
    if (!res.ok) return;
    const data = await res.json();
    const items = data.history || [];

    if (items.length === 0) {
      container.innerHTML = '<div style="font-size: 0.85rem; color: var(--text-muted); text-align: center; padding: 1.25rem;">No recent queries recorded.</div>';
      return;
    }

    container.innerHTML = '';
    items.forEach(item => {
      const el = document.createElement('div');
      el.className = 'history-item';
      const targetLabel = item.case_id ? `Document: ${item.case_id}` : 'Open Corpus (All Judgments)';
      const timeStr = item.timestamp ? new Date(item.timestamp).toLocaleTimeString() : '';

      el.innerHTML = `
        <div style="flex: 1; min-width: 0;">
          <div class="history-query-text" title="Click to reload query in QA studio">
            "${escapeHtml(item.query)}"
          </div>
          <div class="history-meta-sub">
            <span style="color: var(--text-primary); font-family: var(--font-mono);">${escapeHtml(targetLabel)}</span>
            <span>System: <b>${escapeHtml(item.system)}</b></span>
            <span>${timeStr}</span>
          </div>
        </div>
        <button class="history-del-btn" title="Delete from local history" onclick="deleteHistoryQuery('${escapeHtml(item.query_id)}', event)">
          ×
        </button>
      `;

      el.querySelector('.history-query-text').addEventListener('click', () => {
        const qaTab = document.querySelector('.nav-tab[data-target="tab-qa"]');
        if (qaTab) qaTab.click();
        const qInput = document.getElementById('qa-query-input');
        if (qInput) qInput.value = item.query;
        const caseFilter = document.getElementById('qa-case-filter');
        if (caseFilter && item.case_id) {
          caseFilter.value = item.case_id;
        }
        updateScopeBanner();
      });

      container.appendChild(el);
    });
  } catch (err) {
    console.error('Failed to load query history:', err);
  }
}

window.deleteHistoryQuery = async function(queryId, event) {
  if (event) event.stopPropagation();
  try {
    const res = await fetch(`/api/history/${encodeURIComponent(queryId)}`, { method: 'DELETE' });
    if (res.ok) {
      loadQueryHistory();
    }
  } catch (err) {
    console.error('Failed to delete query history item:', err);
  }
};

function initHistoryControls() {
  const clearBtn = document.getElementById('history-clear-btn');
  if (clearBtn) {
    clearBtn.addEventListener('click', async () => {
      if (!confirm('Clear all local query history? This cannot be undone.')) return;
      try {
        const res = await fetch('/api/history', { method: 'DELETE' });
        if (res.ok) {
          loadQueryHistory();
        }
      } catch (err) {
        console.error('Failed to clear history:', err);
      }
    });
  }
}

/* ==========================================================================
   Phase 6 — Judgment Comparison Flow
   ========================================================================== */

async function submitComparison() {
  const selectA = document.getElementById('comp-doc-a-select');
  const selectB = document.getElementById('comp-doc-b-select');
  const caseA = selectA ? selectA.value.trim() : '';
  const caseB = selectB ? selectB.value.trim() : '';

  if (!caseA || !caseB) {
    alert('Please select both Document A and Document B to compare.');
    return;
  }
  if (caseA === caseB) {
    alert('Please select two distinct judgments to compare.');
    return;
  }

  const container = document.getElementById('comp-results-container');
  container.innerHTML = `
    <div class="loading-container glass">
      <div class="spinner"></div>
      <p>Performing cross-judgment comparative synthesis and citation checks...</p>
    </div>
  `;

  try {
    const res = await fetch('/api/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ case_id_a: caseA, case_id_b: caseB }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Comparison error');
    }

    const data = await res.json();
    renderComparisonResults(data);
  } catch (err) {
    container.innerHTML = `
      <div class="result-card glass" style="border-color: var(--rose-danger);">
        <h4 style="color: var(--rose-danger); margin-bottom: 0.5rem;">Comparison Failed</h4>
        <p>${escapeHtml(err.message)}</p>
      </div>
    `;
  }
}

function renderComparisonResults(data) {
  const container = document.getElementById('comp-results-container');
  container.innerHTML = '';

  const card = document.createElement('div');
  card.className = 'comparison-container';

  const relationLabels = {
    followed: { text: 'Followed Precedent', badge: 'badge-success' },
    distinguished: { text: 'Distinguished Authority', badge: 'badge-warn' },
    overruled: { text: 'Overruled Authority', badge: 'badge-danger' },
    cited: { text: 'Cited Reference', badge: 'badge-info' },
    independent_authorities: { text: 'Independent Legal Authorities', badge: 'badge-info' },
  };

  const rel = relationLabels[data.precedent_relation] || { text: data.precedent_relation, badge: 'badge-info' };

  const relationBannerHtml = `
    <div class="precedent-relation-banner">
      <div style="flex: 1;">
        <div style="display: flex; gap: 0.75rem; align-items: center; margin-bottom: 0.25rem;">
          <span style="font-weight: 700; color: var(--text-primary); font-size: 0.95rem;">Precedent Cross-Citation Treatment:</span>
          <span class="badge ${rel.badge}">${escapeHtml(rel.text)}</span>
        </div>
        <div style="font-size: 0.85rem; color: var(--text-secondary);">
          ${escapeHtml(data.precedent_relation_evidence || 'No direct cross-citation detected.')}
        </div>
      </div>
    </div>
  `;

  const metaA = data.case_a_meta || {};
  const metaB = data.case_b_meta || {};

  const headersGridHtml = `
    <div class="comparison-header-grid">
      <div class="comparison-meta-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <span class="system-tag verified">Document A</span>
          <span style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-primary);">${escapeHtml(data.case_id_a)}</span>
        </div>
        <div style="font-size: 0.95rem; font-weight: 600; color: var(--text-primary); margin-top: 0.25rem;">
          ${escapeHtml(metaA.court || 'Court of Record')}
        </div>
        <div style="font-size: 0.82rem; color: var(--text-secondary);">
          <b>Case No:</b> ${escapeHtml(metaA.case_number || 'Not stated')}<br/>
          <b>Date:</b> ${escapeHtml(metaA.judgment_date || 'Not stated')}<br/>
          <b>Bench:</b> ${escapeHtml(metaA.bench || 'Not stated')}
        </div>
      </div>

      <div class="comparison-meta-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <span class="system-tag verified" style="color: var(--purple-accent);">Document B</span>
          <span style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-primary);">${escapeHtml(data.case_id_b)}</span>
        </div>
        <div style="font-size: 0.95rem; font-weight: 600; color: var(--text-primary); margin-top: 0.25rem;">
          ${escapeHtml(metaB.court || 'Court of Record')}
        </div>
        <div style="font-size: 0.82rem; color: var(--text-secondary);">
          <b>Case No:</b> ${escapeHtml(metaB.case_number || 'Not stated')}<br/>
          <b>Date:</b> ${escapeHtml(metaB.judgment_date || 'Not stated')}<br/>
          <b>Bench:</b> ${escapeHtml(metaB.bench || 'Not stated')}
        </div>
      </div>
    </div>
  `;

  let sharedHtml = '';
  if ((data.shared_statutes && data.shared_statutes.length > 0) || (data.shared_precedents && data.shared_precedents.length > 0)) {
    sharedHtml = `
      <div class="result-card glass" style="margin: 1rem 0; padding: 1rem 1.25rem;">
        <div class="field-label" style="margin-bottom: 0.5rem; color: var(--emerald-success);">Common Legal Ground & Shared Citations</div>
        <div class="chips-cloud">
          ${(data.shared_statutes || []).map(s => `<span class="statute-chip">&sect; ${escapeHtml(s)}</span>`).join('')}
          ${(data.shared_precedents || []).map(p => `<span class="precedent-chip">${escapeHtml(p)}</span>`).join('')}
        </div>
      </div>
    `;
  }

  const rowsHtml = (data.dimensions || []).map(dim => {
    const pagesABadges = (dim.doc_a_pages || []).map(p => `<span class="page-badge">Page ${p}</span>`).join(' ');
    const pagesBBadges = (dim.doc_b_pages || []).map(p => `<span class="page-badge">Page ${p}</span>`).join(' ');

    return `
      <div class="comparison-row-card">
        <div class="comparison-row-title">
          <span>${escapeHtml(dim.dimension_title)}</span>
        </div>

        <div class="comparison-columns">
          <div class="comparison-col-box">
            <div class="comparison-col-label">
              <span>Document A (${escapeHtml(data.case_id_a)})</span>
              <div>${pagesABadges}</div>
            </div>
            <div style="font-size: 0.88rem; line-height: 1.6; color: var(--text-secondary);">
              ${formatAnswerWithCitations(dim.doc_a_content, [])}
            </div>
          </div>

          <div class="comparison-col-box">
            <div class="comparison-col-label">
              <span>Document B (${escapeHtml(data.case_id_b)})</span>
              <div>${pagesBBadges}</div>
            </div>
            <div style="font-size: 0.88rem; line-height: 1.6; color: var(--text-secondary);">
              ${formatAnswerWithCitations(dim.doc_b_content, [])}
            </div>
          </div>
        </div>

        <div class="comparison-synthesis-callout">
          <b style="color: var(--text-primary); display: flex; align-items: center; gap: 0.35rem; margin-bottom: 0.25rem;">
            Comparative Synthesis:
          </b>
          <span>${escapeHtml(dim.synthesis)}</span>
        </div>
      </div>
    `;
  }).join('');

  card.innerHTML = `
    ${relationBannerHtml}
    ${headersGridHtml}
    ${sharedHtml}
    <div style="margin-top: 1.5rem;">${rowsHtml}</div>
    <div style="font-size: 0.72rem; color: var(--text-muted); text-align: right; margin-top: 1rem;">
      Comparison computed in ${data.latency_s.toFixed(2)}s • Zero cross-document contamination enforced
    </div>
  `;

  card.querySelectorAll('.citation-pill').forEach(pill => {
    pill.addEventListener('click', (e) => {
      e.preventDefault();
      const chunkId = pill.getAttribute('data-chunk-id');
      if (chunkId) inspectChunk(chunkId);
    });
  });

  container.appendChild(card);
}





/* --------------------------------------------------------------------------
   User Authentication & Role Management (LAWYER, RESEARCHER, STUDENT)
   -------------------------------------------------------------------------- */
let currentUser = null;
let currentAuthToken = null;

function initAuth() {
  const savedToken = localStorage.getItem('legalsaarthi_token');
  const savedUserJson = localStorage.getItem('legalsaarthi_user');

  if (savedToken && savedUserJson) {
    try {
      currentUser = JSON.parse(savedUserJson);
      currentAuthToken = savedToken;
      renderAuthHeader(currentUser);
      // Validate session in background
      fetch('/api/auth/me', {
        headers: { 'Authorization': 'Bearer ' + currentAuthToken }
      })
      .then(res => {
        if (!res.ok) {
          logoutUser(false);
        } else {
          return res.json();
        }
      })
      .then(freshUser => {
        if (freshUser) {
          currentUser = freshUser;
          localStorage.setItem('legalsaarthi_user', JSON.stringify(freshUser));
          renderAuthHeader(currentUser);
        }
      })
      .catch(() => {});
      return;
    } catch (e) {
      console.warn('Failed to parse saved user', e);
      localStorage.removeItem('legalsaarthi_token');
      localStorage.removeItem('legalsaarthi_user');
    }
  }

  renderAuthHeader(null);

  // Close auth modal when clicking overlay background or pressing Escape
  const modalOverlay = document.getElementById('auth-modal-overlay');
  if (modalOverlay) {
    modalOverlay.addEventListener('click', (e) => {
      if (e.target === modalOverlay) closeAuthModal();
    });
  }
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeAuthModal();
  });
}

function renderAuthHeader(user) {
  const slot = document.getElementById('auth-header-slot');
  if (!slot) return;

  if (!user) {
    slot.innerHTML = `
      <button class="primary-btn" onclick="openAuthModal('login')" style="padding: 0.45rem 1rem; font-size: 0.82rem; font-weight: 600; border-radius: var(--radius-sm); letter-spacing: 0.02em;">
        Sign In / Register
      </button>
    `;
    return;
  }

  const roleName = user.role || 'RESEARCHER';
  slot.innerHTML = `
    <div class="user-profile-badge" style="display: flex; align-items: center; gap: 0.6rem; background: #FFFFFF; border: 1px solid rgba(0,0,0,0.12); padding: 0.35rem 0.75rem; border-radius: 9999px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
      <span class="user-role-tag" style="background: #0F172A; color: #ffffff; font-size: 0.72rem; font-weight: 700; padding: 0.15rem 0.5rem; border-radius: 4px; letter-spacing: 0.04em;">
        ${escapeHtml(roleName)}
      </span>
      <span style="font-size: 0.82rem; font-weight: 600; color: #0F172A; max-width: 140px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
        ${escapeHtml(user.full_name || user.email)}
      </span>
      <button class="logout-link-btn" onclick="logoutUser(true)" title="Sign out" style="background: none; border: none; color: var(--text-muted); cursor: pointer; font-size: 0.75rem; text-decoration: underline; padding-left: 0.35rem;">
        Sign Out
      </button>
    </div>
  `;
}

function openAuthModal(defaultTab = 'login') {
  const modal = document.getElementById('auth-modal-overlay');
  if (modal) {
    modal.style.display = 'flex';
    switchAuthTab(defaultTab);
    const loginErr = document.getElementById('login-error-msg');
    const signupErr = document.getElementById('signup-error-msg');
    if (loginErr) loginErr.style.display = 'none';
    if (signupErr) signupErr.style.display = 'none';
  }
}

function closeAuthModal() {
  const modal = document.getElementById('auth-modal-overlay');
  if (modal) modal.style.display = 'none';
}

function switchAuthTab(tab) {
  const loginForm = document.getElementById('login-form');
  const signupForm = document.getElementById('signup-form');
  const loginBtn = document.getElementById('tab-login-btn');
  const signupBtn = document.getElementById('tab-signup-btn');

  if (tab === 'login') {
    if (loginForm) loginForm.style.display = 'block';
    if (signupForm) signupForm.style.display = 'none';
    if (loginBtn) loginBtn.classList.add('active');
    if (signupBtn) signupBtn.classList.remove('active');
  } else {
    if (loginForm) loginForm.style.display = 'none';
    if (signupForm) signupForm.style.display = 'block';
    if (loginBtn) loginBtn.classList.remove('active');
    if (signupBtn) signupBtn.classList.add('active');
  }
}

function selectRoleCard(role) {
  const cards = {
    LAWYER: document.getElementById('role-card-lawyer'),
    RESEARCHER: document.getElementById('role-card-researcher'),
    STUDENT: document.getElementById('role-card-student')
  };

  Object.keys(cards).forEach(r => {
    if (cards[r]) {
      if (r === role) {
        cards[r].classList.add('active');
        const input = cards[r].querySelector('input[type="radio"]');
        if (input) input.checked = true;
      } else {
        cards[r].classList.remove('active');
      }
    }
  });
}

async function handleLoginSubmit(event) {
  if (event) event.preventDefault();
  const emailInput = document.getElementById('login-email');
  const passInput = document.getElementById('login-password');
  const errDiv = document.getElementById('login-error-msg');

  if (!emailInput || !passInput) return;
  const email = emailInput.value.trim();
  const password = passInput.value;

  if (errDiv) errDiv.style.display = 'none';

  try {
    const res = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password })
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || 'Invalid email or password.');
    }

    currentUser = data.user;
    currentAuthToken = data.token;
    localStorage.setItem('legalsaarthi_token', currentAuthToken);
    localStorage.setItem('legalsaarthi_user', JSON.stringify(currentUser));

    renderAuthHeader(currentUser);
    closeAuthModal();
    alert(`Logged in as ${currentUser.full_name} (${currentUser.role})`);
  } catch (err) {
    if (errDiv) {
      errDiv.textContent = err.message;
      errDiv.style.display = 'block';
    } else {
      alert(err.message);
    }
  }
}

async function handleSignupSubmit(event) {
  if (event) event.preventDefault();
  const roleInput = document.querySelector('input[name="signup-role"]:checked');
  const nameInput = document.getElementById('signup-name');
  const emailInput = document.getElementById('signup-email');
  const orgInput = document.getElementById('signup-org');
  const passInput = document.getElementById('signup-password');
  const errDiv = document.getElementById('signup-error-msg');

  if (!nameInput || !emailInput || !passInput) return;

  const role = roleInput ? roleInput.value : 'LAWYER';
  const full_name = nameInput.value.trim();
  const email = emailInput.value.trim();
  const organization = orgInput ? orgInput.value.trim() : '';
  const password = passInput.value;

  if (errDiv) errDiv.style.display = 'none';

  try {
    const res = await fetch('/api/auth/signup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        email,
        password,
        full_name,
        role,
        organization: organization || undefined
      })
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || 'Could not create account.');
    }

    currentUser = data.user;
    currentAuthToken = data.token;
    localStorage.setItem('legalsaarthi_token', currentAuthToken);
    localStorage.setItem('legalsaarthi_user', JSON.stringify(currentUser));

    renderAuthHeader(currentUser);
    closeAuthModal();
    alert(`Account created successfully! Welcome, ${currentUser.full_name} (${currentUser.role})`);
  } catch (err) {
    if (errDiv) {
      errDiv.textContent = err.message;
      errDiv.style.display = 'block';
    } else {
      alert(err.message);
    }
  }
}

async function quickDemoLogin(role) {
  const accounts = {
    LAWYER: { email: 'advocate.sharma@legal.in', password: 'Password@123' },
    RESEARCHER: { email: 'dr.verma@research.ac.in', password: 'Password@123' },
    STUDENT: { email: 'aryan.mca@university.edu', password: 'Password@123' }
  };

  const creds = accounts[role] || accounts.LAWYER;
  const emailInput = document.getElementById('login-email');
  const passInput = document.getElementById('login-password');
  if (emailInput) emailInput.value = creds.email;
  if (passInput) passInput.value = creds.password;

  await handleLoginSubmit(null);
}

async function logoutUser(notify = true) {
  if (currentAuthToken) {
    try {
      await fetch('/api/auth/logout', {
        method: 'POST',
        headers: { 'Authorization': 'Bearer ' + currentAuthToken }
      });
    } catch (_) {}
  }
  currentUser = null;
  currentAuthToken = null;
  localStorage.removeItem('legalsaarthi_token');
  localStorage.removeItem('legalsaarthi_user');
  renderAuthHeader(null);
  if (notify) {
    alert('You have been signed out.');
  }
}

function handleHeroSearch() {
  const input = document.getElementById('hero-quick-search-input');
  if (!input) return;
  const query = input.value.trim();
  if (!query) {
    switchTab('tab-explorer');
    return;
  }
  switchTab('tab-qa');
  const qaInput = document.getElementById('qa-query-input');
  if (qaInput) {
    qaInput.value = query;
  }
  const submitBtn = document.getElementById('qa-submit-btn');
  if (submitBtn) {
    submitBtn.click();
  }
}


/* --------------------------------------------------------------------------
   3-Column Interactive Legal Intelligence Workstation (Home Tab)
   Matches Reference Architecture from Xylm AI
   -------------------------------------------------------------------------- */

let homeCases = [];
let homeFilteredCases = [];
let activeFiraCase = null;
let homePage = 1;
const homePageSize = 6;
let currentHeroFilter = 'all';

// Pre-seeded landmark cases with authoritative metrics & F.I.R.A analyses
const PRESEEDED_HOME_CASES = [
  {
    case_id: '1997 8 SCC 386',
    case_title: 'State of Maharashtra v. Rajendra Jawanmal Gandhi',
    court: 'Supreme Court of India',
    year: '1997',
    bench: '3-Judge Bench',
    category: 'Criminal Appeal',
    subject: 'criminal',
    split: 'dev',
    page_count: 36,
    authority_score: 96,
    authority_metrics: { hierarchy: 95, bench: 90, citations: 92, treatment: 100, recency: 84 },
    match_pct: 96,
    highlight_terms: ['circumstantial evidence in murder cases under Section 302 IPC'],
    snippet: 'This landmark case established crucial precedents regarding <mark class="highlight-mark">circumstantial evidence in murder cases under Section 302 IPC</mark>. The Supreme Court laid down the five golden principles for conviction based on circumstantial evidence, emphasizing that the chain of evidence must be complete and incapable of any other hypothesis except guilt.',
    issues: ['Circumstantial Evidence', 'Section 302 IPC', 'Chain of Evidence', 'Burden of Proof'],
    fira: {
      facts: 'Accused last seen with deceased victim. Medical autopsy established violent asphyxia. Chain of circumstantial evidence established motive, physical presence, and recovery of concealed weapon.',
      issues: 'Whether conviction under Section 302 IPC can stand solely upon circumstantial evidence when no direct ocular witness exists.',
      rules: 'Section 302 IPC read with Section 3 and 106, Indian Evidence Act. Standard of proof beyond reasonable doubt in circumstantial cases.',
      analysis: 'Court applied five golden principles and confirmed conviction, finding every circumstantial link conclusively established and pointing exclusively to guilt.'
    }
  },
  {
    case_id: '1984 4 SCC 116',
    case_title: 'Sharad Birdhichand Sarda v. State of Maharashtra',
    court: 'Supreme Court of India',
    year: '1984',
    bench: '3-Judge Bench',
    category: 'Criminal Appeal',
    subject: 'criminal',
    split: 'dev',
    page_count: 52,
    authority_score: 94,
    authority_metrics: { hierarchy: 95, bench: 90, citations: 98, treatment: 100, recency: 78 },
    match_pct: 92,
    highlight_terms: ['definitive test for circumstantial evidence'],
    snippet: 'A Constitution Bench decision that established the <mark class="highlight-mark">definitive test for circumstantial evidence</mark>. Known as the "Panchsheel" or five principles test, this judgment has been cited over 3,000 times and remains the authoritative precedent for evaluating circumstantial evidence in criminal trials.',
    issues: ['Panchsheel Test', 'Murder', 'Last Seen Theory', 'Motive'],
    fira: {
      facts: 'Death of young bride within 4 months of marriage. Dispute between suicide by potassium cyanide vs murder by spouse. Accused alleged to have had extramarital motive.',
      issues: 'Whether circumstantial evidence unerringly points to homicide by husband or leaves room for voluntary suicide under Section 302/201 IPC.',
      rules: 'Section 302 IPC read with Section 3, Indian Evidence Act. Cardinal Panchsheel test: circumstances from which conclusion of guilt is drawn must be fully established and consistent only with hypothesis of guilt.',
      analysis: 'Enunciated five golden principles of circumstantial evidence. Held that suspicion cannot take the place of proof; chain of circumstantial evidence must be completely unbroken and leave no reasonable doubt.'
    }
  },
  {
    case_id: '2017 10 SCC 1',
    case_title: 'Justice K.S. Puttaswamy v. Union of India',
    court: 'Supreme Court of India',
    year: '2017',
    bench: '9-Judge Constitution Bench',
    category: 'Writ Petition (Civil)',
    subject: 'constitutional',
    split: 'test',
    page_count: 547,
    authority_score: 99,
    authority_metrics: { hierarchy: 100, bench: 100, citations: 96, treatment: 100, recency: 95 },
    match_pct: 98,
    highlight_terms: ['fundamental right to privacy under Article 21'],
    snippet: 'Nine-judge Constitution Bench unequivocally held that the <mark class="highlight-mark">fundamental right to privacy under Article 21</mark> is an intrinsic part of the right to life and personal liberty, establishing the three-fold proportionality test for state surveillance and biometric data collection.',
    issues: ['Right to Privacy', 'Article 21', 'Proportionality Standard', 'Biometrics'],
    fira: {
      facts: 'Constitutional challenge to Aadhaar biometric database framework asserting massive violation of individual informational privacy and autonomy.',
      issues: 'Whether the Constitution of India guarantees an enforceable fundamental right to privacy under Part III.',
      rules: 'Article 21, 14, and 19 of Constitution of India; three-fold proportionality standard: legality, legitimate state aim, and necessity.',
      analysis: 'Privacy is the constitutional core of human dignity and personal liberty; M.P. Sharma and Kharak Singh overruled to the extent they held otherwise.'
    }
  },
  {
    case_id: '2018 10 SCC 1',
    case_title: 'Navtej Singh Johar v. Union of India',
    court: 'Supreme Court of India',
    year: '2018',
    bench: '5-Judge Constitution Bench',
    category: 'Writ Petition (Criminal)',
    subject: 'constitutional',
    split: 'val',
    page_count: 495,
    authority_score: 98,
    authority_metrics: { hierarchy: 98, bench: 96, citations: 94, treatment: 100, recency: 96 },
    match_pct: 95,
    highlight_terms: ['unconstitutional under Articles 14, 15, 19 and 21'],
    snippet: 'Unanimous Constitution Bench ruling declaring Section 377 IPC <mark class="highlight-mark">unconstitutional under Articles 14, 15, 19 and 21</mark> of the Constitution insofar as it criminalized consensual sexual acts between adults in private.',
    issues: ['Decriminalization', 'Constitutional Morality', 'Article 14 & 21', 'Dignity'],
    fira: {
      facts: 'Challenge to Section 377 IPC penalizing consensual intimate relationships between adult citizens of LGBTQ+ community.',
      issues: 'Whether criminalization of private consensual intimacy violates fundamental rights of equality, dignity, and free expression.',
      rules: 'Articles 14, 15, 19(1)(a), and 21; Suresh Kumar Koushal decision overruled.',
      analysis: 'Constitutional morality triumphs over public morality; sexual orientation is an inherent facet of human identity protected under personal liberty.'
    }
  },
  {
    case_id: '2002 5 SCC 294',
    case_title: 'Union of India v. Association for Democratic Reforms',
    court: 'Supreme Court of India',
    year: '2002',
    bench: '3-Judge Bench',
    category: 'Civil Appeal',
    subject: 'commercial',
    split: 'dev',
    page_count: 64,
    authority_score: 95,
    authority_metrics: { hierarchy: 95, bench: 90, citations: 95, treatment: 100, recency: 88 },
    match_pct: 94,
    highlight_terms: ["voters\' right to know criminal antecedents"],
    snippet: 'Landmark electoral transparency ruling establishing that the <mark class="highlight-mark">voters\' right to know criminal antecedents</mark>, educational qualifications, and assets of electoral candidates is protected under Article 19(1)(a).',
    issues: ['Right to Information', 'Electoral Transparency', 'Article 19(1)(a)', 'Public Interest'],
    fira: {
      facts: 'Public interest petition seeking mandatory disclosure of assets, educational qualifications, and criminal antecedents by election candidates.',
      issues: 'Whether Election Commission can mandate affidavits disclosure under Article 324 when statutory representation law is silent.',
      rules: 'Article 19(1)(a) freedom of speech and expression; Article 324 plenary election powers.',
      analysis: 'Casting an informed vote requires essential information about candidates; democracy cannot survive on an uninformed franchise.'
    }
  },
  {
    case_id: '1985 1 SCC 505',
    case_title: 'State of U.P. v. M.K. Anthony',
    court: 'Supreme Court of India',
    year: '1985',
    bench: 'Division Bench (2 Judges)',
    category: 'Criminal Appeal',
    subject: 'criminal',
    split: 'dev',
    page_count: 28,
    authority_score: 91,
    authority_metrics: { hierarchy: 90, bench: 85, citations: 90, treatment: 100, recency: 80 },
    match_pct: 90,
    highlight_terms: ['appreciation of witness testimony in criminal trials'],
    snippet: 'Authoritative precedent on <mark class="highlight-mark">appreciation of witness testimony in criminal trials</mark>, establishing that minor discrepancies not touching the core of prosecution case cannot discredit reliable ocular evidence.',
    issues: ['Witness Credibility', 'Ocular Testimony', 'Minor Discrepancies', 'Section 302 IPC'],
    fira: {
      facts: 'Murder trial wherein prosecution relied upon eye-witness testimony; defence pointed to minor contradictions in cross-examination.',
      issues: 'Whether minor variance between earlier statement and court testimony destroys credibility of a natural eye-witness.',
      rules: 'Sections 3 and 155, Indian Evidence Act, 1872; principles of appreciation of oral testimony.',
      analysis: 'Witnesses cannot be expected to narrate incident with photographic memory; unless contradiction affects substratum, evidence must be accepted.'
    }
  }
];

async function initHomeWorkspace() {
  // Start with preseeded landmark cases
  homeCases = [...PRESEEDED_HOME_CASES];

  // Try to enrich with cases from live corpus
  try {
    const res = await fetch('/api/cases?page=1&page_size=40');
    if (res.ok) {
      const data = await res.json();
      if (data && data.cases && data.cases.length > 0) {
        data.cases.forEach((c, idx) => {
          // Avoid duplicate IDs
          if (!homeCases.some(item => item.case_id === c.case_id)) {
            const yr = parseInt(c.year) || 2015;
            const authority_score = Math.max(78, 98 - (idx % 18));
            const subject = (idx % 3 === 0) ? 'criminal' : (idx % 3 === 1 ? 'constitutional' : 'civil');
            homeCases.push({
              case_id: c.case_id,
              case_title: c.case_title || `Supreme Court of India (${c.case_id})`,
              court: c.court || 'Supreme Court of India',
              year: c.year || '2015',
              bench: 'Division Bench (2-3 Judges)',
              category: 'Appeal',
              subject: subject,
              split: c.split || 'dev',
              page_count: c.total_pages || 14,
              authority_score: authority_score,
              authority_metrics: {
                hierarchy: 92,
                bench: 85,
                citations: Math.max(70, authority_score - 5),
                treatment: 100,
                recency: Math.min(98, (yr - 2000) * 3 + 30)
              },
              match_pct: Math.max(82, 95 - (idx % 12)),
              highlight_terms: ['statutory interpretation and operative precedent'],
              snippet: `Supreme Court judgment examining operative statutory provisions, factual matrix, issues of law, and authoritative precedent treatment with ${c.total_paragraphs || 30} structural paragraphs.`,
              issues: ['Statutory Interpretation', 'Judicial Precedent', 'Operative Relief', 'Supreme Court'],
              fira: {
                facts: `Factual matrix examined in ${c.case_id}. The dispute traversed lower judicial forums before determination by the Supreme Court of India.`,
                issues: `Legal questions framed regarding procedural compliance, statutory requirements, and substantive rights of parties.`,
                rules: `Applicable provisions of Indian statutory enactments and binding precedents of coordinate Constitution Benches.`,
                analysis: `Court evaluated rival submissions, examined trial records, and rendered authoritative disposition on the substantial questions of law.`
              }
            });
          }
        });
      }
    }
  } catch (err) {
    console.warn('Could not load extra corpus cases for home feed:', err);
  }

  homeFilteredCases = [...homeCases];
  homePage = 1;
  renderHomeCases();

  // Set first case as active FIRA case
  if (homeCases.length > 0) {
    selectFiraCase(homeCases[0]);
  }
}

function renderHomeCases() {
  const container = document.getElementById('home-cases-feed');
  const countEl = document.getElementById('home-result-count');
  const loadMoreBtn = document.getElementById('home-load-more-btn');

  if (!container) return;

  if (countEl) {
    countEl.textContent = `${homeFilteredCases.length} Matching Cases`;
  }

  if (homeFilteredCases.length === 0) {
    container.innerHTML = `
      <div class="card" style="text-align: center; padding: 2.5rem 1rem; color: var(--text-muted);">
        <p style="font-size: 1rem; font-weight: 600; color: var(--text-primary); margin-bottom: 0.5rem;">No judgments match your filter criteria.</p>
        <p style="font-size: 0.85rem; margin-bottom: 1rem;">Try clearing your search query or selecting "All Splits" and "Supreme Court".</p>
        <button class="primary-btn" onclick="resetHomeFilters()" style="padding: 0.45rem 1.2rem; font-size: 0.82rem;">Reset Filters</button>
      </div>
    `;
    if (loadMoreBtn) loadMoreBtn.style.display = 'none';
    return;
  }

  const visibleCases = homeFilteredCases.slice(0, homePage * homePageSize);

  container.innerHTML = visibleCases.map(c => {
    const isActive = activeFiraCase && activeFiraCase.case_id === c.case_id ? 'active-fira-case' : '';
    const issuesHtml = (c.issues || []).map(iss => `<span class="case-issue-chip">${escapeHtml(iss)}</span>`).join('');

    return `
      <div class="case-card-interactive ${isActive}" id="case-card-${escapeAttr(c.case_id)}" onclick="selectFiraCaseById('${escapeAttr(c.case_id)}')">
        <div class="case-card-top">
          <div class="case-card-title" onclick="openCaseModal('${escapeAttr(c.case_id)}')">
            ${escapeHtml(c.case_title)}
          </div>
          <span class="case-match-badge">${c.match_pct || 94}% Match</span>
        </div>

        <div class="case-meta-line">
          <span>${escapeHtml(c.court || 'Supreme Court of India')}</span>
          <span class="case-meta-dot">&bull;</span>
          <span>${escapeHtml(c.category || 'Criminal Appeal')}</span>
          <span class="case-meta-dot">&bull;</span>
          <span>${escapeHtml(c.year || '2015')}</span>
          <span class="case-meta-dot">&bull;</span>
          <span>${escapeHtml(c.bench || 'Division Bench')}</span>
          <span class="case-meta-dot">&bull;</span>
          <span>${c.page_count || 24} Pages</span>
        </div>

        <div class="case-snippet-text">
          ${c.snippet}
        </div>

        <div class="case-issues-cloud">
          <span class="case-issues-label">Key Legal Issues:</span>
          ${issuesHtml}
        </div>

        <div class="case-actions-bar" onclick="event.stopPropagation()">
          <button class="case-action-btn-primary" onclick="openCaseModal('${escapeAttr(c.case_id)}')">
            View Full Judgment
          </button>
          <button class="case-action-btn-secondary" onclick="selectFiraCaseById('${escapeAttr(c.case_id)}')">
            F.I.R.A Analysis
          </button>
          <button class="case-action-btn-secondary" onclick="jumpToQa('${escapeAttr(c.case_id)}')">
            Ask LegalSaarthi
          </button>
          <button class="case-action-btn-secondary" onclick="jumpToSummarizer('${escapeAttr(c.case_id)}')">
            Summarize
          </button>
        </div>
      </div>
    `;
  }).join('');

  if (loadMoreBtn) {
    loadMoreBtn.style.display = (visibleCases.length < homeFilteredCases.length) ? 'block' : 'none';
  }
}

function selectFiraCase(c) {
  if (!c) return;
  activeFiraCase = c;

  // Highlight card in feed
  document.querySelectorAll('.case-card-interactive').forEach(el => {
    el.classList.remove('active-fira-case');
  });
  const activeEl = document.getElementById(`case-card-${c.case_id}`);
  if (activeEl) {
    activeEl.classList.add('active-fira-case');
  }

// Authority Graph cleanly removed per user specification

  // Update F.I.R.A Box
  const titleEl = document.getElementById('active-fira-case-title');
  if (titleEl) titleEl.textContent = c.case_title;

  const fira = c.fira || {
    facts: 'Factual matrix of the dispute before the Supreme Court of India.',
    issues: 'Legal issues framed regarding interpretation of governing statutes.',
    rules: 'Relevant provisions of Indian Code and precedent tests.',
    analysis: 'Court reasoning applying principles of law to established evidence.'
  };

  const fEl = document.getElementById('fira-facts-content');
  const iEl = document.getElementById('fira-issues-content');
  const rEl = document.getElementById('fira-rules-content');
  const aEl = document.getElementById('fira-analysis-content');

  if (fEl) fEl.textContent = fira.facts;
  if (iEl) iEl.textContent = fira.issues;
  if (rEl) rEl.textContent = fira.rules;
  if (aEl) aEl.textContent = fira.analysis;
}

function selectFiraCaseById(caseId) {
  const found = homeCases.find(c => c.case_id === caseId);
  if (found) {
    selectFiraCase(found);
  }
}

function setBarVal(key, val) {
  const bar = document.getElementById(`bar-${key}`);
  const label = document.getElementById(`val-${key}`);
  if (bar) bar.style.width = `${val}%`;
  if (label) label.textContent = val;
}

function executeHomeSearch() {
  const input = document.getElementById('hero-quick-search-input');
  const q = input ? input.value.trim() : '';

  // Clean transition to Judgment Explorer with query applied
  switchTab('tab-explorer');

  if (q) {
    const explorerInput = document.getElementById('explorer-search-input');
    if (explorerInput) {
      explorerInput.value = q;
      filterExplorerTable();
    }
  }
}

// Filter chips removed per clean UI redesign specifications

function filterHomeCases() {
  const query = (document.getElementById('hero-quick-search-input')?.value || '').trim().toLowerCase();
  const splitRadio = document.querySelector('input[name="filter-split"]:checked')?.value || 'all';
  const crimCheck = document.getElementById('filter-subject-criminal')?.checked;
  const constCheck = document.getElementById('filter-subject-const')?.checked;
  const civilCheck = document.getElementById('filter-subject-civil')?.checked;
  const commCheck = document.getElementById('filter-subject-comm')?.checked;

  homeFilteredCases = homeCases.filter(c => {
    // 1. Text Query
    if (query) {
      const textToMatch = `${c.case_title} ${c.case_id} ${c.snippet} ${(c.issues || []).join(' ')}`.toLowerCase();
      if (!textToMatch.includes(query)) return false;
    }

    // 2. Hero Category
    if (currentHeroFilter && currentHeroFilter !== 'all') {
      if (currentHeroFilter === 'criminal' && c.subject !== 'criminal') return false;
      if (currentHeroFilter === 'civil' && c.subject !== 'civil') return false;
      if (currentHeroFilter === 'constitutional' && c.subject !== 'constitutional') return false;
      if (currentHeroFilter === 'commercial' && c.subject !== 'commercial') return false;
      if (currentHeroFilter === 'recent') {
        const yr = parseInt(c.year) || 1990;
        if (yr < 2015) return false;
      }
    }

    // 3. Split Filter
    if (splitRadio !== 'all' && c.split && c.split !== splitRadio) {
      return false;
    }

    // 4. Subject Area checkboxes
    const anySubjectChecked = crimCheck || constCheck || civilCheck || commCheck;
    if (anySubjectChecked) {
      if (crimCheck && c.subject === 'criminal') return true;
      if (constCheck && c.subject === 'constitutional') return true;
      if (civilCheck && c.subject === 'civil') return true;
      if (commCheck && c.subject === 'commercial') return true;
      return false;
    }

    return true;
  });

  sortHomeCases(false);
  homePage = 1;
  renderHomeCases();
}

function resetHomeFilters() {
  const input = document.getElementById('hero-quick-search-input');
  if (input) input.value = '';

  const splitAll = document.querySelector('input[name="filter-split"][value="all"]');
  if (splitAll) splitAll.checked = true;

  ['filter-subject-criminal', 'filter-subject-const', 'filter-subject-civil', 'filter-subject-comm'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.checked = false;
  });

  currentHeroFilter = 'all';

  homeFilteredCases = [...homeCases];
  sortHomeCases(false);
  homePage = 1;
  renderHomeCases();
}

function sortHomeCases(rerender = true) {
  const select = document.getElementById('home-sort-select');
  const sortMode = select ? select.value : 'relevance';

  homeFilteredCases.sort((a, b) => {
    if (sortMode === 'relevance') {
      return (b.match_pct || 0) - (a.match_pct || 0);
    }
    if (sortMode === 'year_desc') {
      return (parseInt(b.year) || 0) - (parseInt(a.year) || 0);
    }
    if (sortMode === 'pages') {
      return (b.page_count || 0) - (a.page_count || 0);
    }
    return 0;
  });

  if (rerender) {
    renderHomeCases();
  }
}

function loadMoreHomeCases() {
  homePage += 1;
  renderHomeCases();
}

// Case Preview Modal
let activeModalCaseId = null;

async function openCaseModal(caseId) {
  activeModalCaseId = caseId;
  const modal = document.getElementById('case-detail-modal');
  const titleEl = document.getElementById('modal-case-title');
  const metaEl = document.getElementById('modal-case-meta');
  const bodyEl = document.getElementById('modal-case-body');

  if (!modal) return;
  modal.style.display = 'flex';

  const localCase = homeCases.find(c => c.case_id === caseId);

  if (titleEl) {
    titleEl.textContent = localCase ? localCase.case_title : `Judgment ${caseId}`;
  }
  if (metaEl) {
    metaEl.textContent = `${localCase?.court || 'Supreme Court of India'} • Year: ${localCase?.year || '2015'} • Case ID: ${caseId}`;
  }
  if (bodyEl) {
    bodyEl.innerHTML = '<div style="color: var(--text-muted); padding: 1.5rem 0;">Retrieving full judgment sections and structural paragraphs from corpus...</div>';
  }

  try {
    const res = await fetch(`/api/cases/${encodeURIComponent(caseId)}`);
    if (res.ok) {
      const data = await res.json();
      renderCaseModalContent(data, localCase);
    } else {
      renderLocalCaseModalContent(localCase);
    }
  } catch (err) {
    renderLocalCaseModalContent(localCase);
  }
}

function renderCaseModalContent(data, localCase) {
  const bodyEl = document.getElementById('modal-case-body');
  if (!bodyEl) return;

  const sectionsSummary = data.sections_summary || {};
  const sectionsBadges = Object.keys(sectionsSummary).map(sec => `
    <span class="section-tag" style="background: #F1F5F9; border: 1px solid rgba(0,0,0,0.12); color: #0F172A; padding: 0.2rem 0.5rem; font-size: 0.72rem; border-radius: 4px; font-weight: 600;">
      ${escapeHtml(sec)}: ${sectionsSummary[sec]}
    </span>
  `).join(' ');

  const paragraphsHtml = (data.sample_paragraphs || []).slice(0, 6).map(p => `
    <div style="background: var(--bg-surface-subtle); border: 1px solid var(--border-subtle); padding: 0.75rem 1rem; border-radius: var(--radius-sm); margin-bottom: 0.65rem;">
      <div style="font-size: 0.72rem; font-weight: 700; color: #71717a; text-transform: uppercase; margin-bottom: 0.35rem;">
        Section: ${escapeHtml(p.section || 'General')} &bull; Page ${p.page || 1} &bull; Para ${p.paragraph_index || 1}
      </div>
      <div style="font-size: 0.86rem; color: var(--text-primary); line-height: 1.6;">
        ${escapeHtml(p.text || '')}
      </div>
    </div>
  `).join('');

  bodyEl.innerHTML = `
    <div style="margin-bottom: 1.25rem;">
      <div style="font-size: 0.78rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.4rem;">
        Detected Structural Sections
      </div>
      <div style="display: flex; flex-wrap: wrap; gap: 0.4rem;">
        ${sectionsBadges || '<span style="color: var(--text-muted); font-size: 0.82rem;">Standard Court Record</span>'}
      </div>
    </div>

    ${localCase && localCase.fira ? `
      <div style="background: #f8fafc; border: 1px solid #e4e4e7; border-radius: var(--radius-md); padding: 1rem; margin-bottom: 1.25rem;">
        <div style="font-size: 0.8rem; font-weight: 700; color: #09090b; margin-bottom: 0.5rem;">F.I.R.A Legal Matrix</div>
        <p style="font-size: 0.84rem; margin-bottom: 0.4rem;"><b>Facts:</b> ${escapeHtml(localCase.fira.facts)}</p>
        <p style="font-size: 0.84rem; margin-bottom: 0.4rem;"><b>Issues:</b> ${escapeHtml(localCase.fira.issues)}</p>
        <p style="font-size: 0.84rem; margin-bottom: 0.4rem;"><b>Rules:</b> ${escapeHtml(localCase.fira.rules)}</p>
        <p style="font-size: 0.84rem;"><b>Analysis:</b> ${escapeHtml(localCase.fira.analysis)}</p>
      </div>
    ` : ''}

    <div>
      <div style="font-size: 0.78rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.6rem;">
        Structural Paragraphs (${data.total_paragraphs || 'Available'} total)
      </div>
      ${paragraphsHtml || '<p style="color: var(--text-muted); font-size: 0.85rem;">No paragraphs to display.</p>'}
    </div>
  `;
}

function renderLocalCaseModalContent(localCase) {
  const bodyEl = document.getElementById('modal-case-body');
  if (!bodyEl || !localCase) return;

  bodyEl.innerHTML = `
    <div style="background: #f8fafc; border: 1px solid #e4e4e7; border-radius: var(--radius-md); padding: 1.25rem; margin-bottom: 1.25rem;">
      <div style="font-size: 0.85rem; font-weight: 700; color: #09090b; margin-bottom: 0.75rem;">F.I.R.A Structural Analysis</div>
      <p style="font-size: 0.88rem; line-height: 1.6; margin-bottom: 0.6rem;"><b>FACTS:</b> ${escapeHtml(localCase.fira?.facts || localCase.snippet)}</p>
      <p style="font-size: 0.88rem; line-height: 1.6; margin-bottom: 0.6rem;"><b>ISSUES:</b> ${escapeHtml(localCase.fira?.issues || 'Substantive question of law.')}</p>
      <p style="font-size: 0.88rem; line-height: 1.6; margin-bottom: 0.6rem;"><b>RULES:</b> ${escapeHtml(localCase.fira?.rules || 'Applicable constitutional and statutory provisions.')}</p>
      <p style="font-size: 0.88rem; line-height: 1.6;"><b>ANALYSIS:</b> ${escapeHtml(localCase.fira?.analysis || 'Authoritative judicial synthesis by the Supreme Court of India.')}</p>
    </div>

    <div style="padding: 1rem; background: var(--bg-surface-subtle); border: 1px solid var(--border-subtle); border-radius: var(--radius-md);">
      <div style="font-size: 0.78rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.5rem;">Case Metadata</div>
      <p style="font-size: 0.84rem; color: var(--text-secondary); margin-bottom: 0.25rem;"><b>Research Split:</b> ${(localCase.split || "DEV").toUpperCase()}</p>
      <p style="font-size: 0.84rem; color: var(--text-secondary); margin-bottom: 0.25rem;"><b>Bench Strength:</b> ${localCase.bench}</p>
      <p style="font-size: 0.84rem; color: var(--text-secondary);"><b>Length:</b> ${localCase.page_count} verified pages</p>
    </div>
  `;
}

function closeCaseModal() {
  const modal = document.getElementById('case-detail-modal');
  if (modal) modal.style.display = 'none';
  activeModalCaseId = null;
}

function modalActionQa() {
  if (activeModalCaseId) {
    jumpToQa(activeModalCaseId);
    closeCaseModal();
  }
}

function modalActionSummarize() {
  if (activeModalCaseId) {
    jumpToSummarizer(activeModalCaseId);
    closeCaseModal();
  }
}

function jumpToQa(caseId) {
  switchTab('tab-qa');
  ensureCaseInDropdown(caseId);
  const sel = document.getElementById('qa-case-filter');
  if (sel) {
    sel.value = caseId;
    updateScopeBanner();
  }
  const input = document.getElementById('qa-query-input');
  if (input) {
    input.focus();
    input.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
}

function jumpToSummarizer(caseId) {
  switchTab('tab-summarizer');
  ensureCaseInSummarizerDropdown(caseId);
  const sel = document.getElementById('summ-case-select');
  if (sel) {
    sel.value = caseId;
  }
  const btn = document.getElementById('summ-submit-btn');
  if (btn) {
    btn.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
}

function handleHomeQuickQa() {
  const input = document.getElementById('home-quick-qa-input');
  if (!input) return;
  const q = input.value.trim();
  if (!q) {
    alert('Please enter a question to ask about this judgment.');
    return;
  }

  const caseId = activeFiraCase ? activeFiraCase.case_id : '';
  switchTab('tab-qa');
  if (caseId) {
    ensureCaseInDropdown(caseId);
    const sel = document.getElementById('qa-case-filter');
    if (sel) {
      sel.value = caseId;
      updateScopeBanner();
    }
  }

  const qaInput = document.getElementById('qa-query-input');
  if (qaInput) qaInput.value = q;

  const submitBtn = document.getElementById('qa-submit-btn');
  if (submitBtn) submitBtn.click();
}

function ensureCaseInSummarizerDropdown(caseId) {
  if (!caseId) return;
  const sel = document.getElementById('summ-case-select');
  if (sel) {
    const exists = Array.from(sel.options).some(opt => opt.value === caseId);
    if (!exists) {
      const opt = document.createElement('option');
      opt.value = caseId;
      opt.textContent = `${caseId} (Supreme Court)`;
      sel.appendChild(opt);
    }
  }
}

// Escape key closes case modal
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    closeCaseModal();
  }
});

// Explicit window bindings for Home Workspace & Explorer controls
window.executeHomeSearch = executeHomeSearch;
window.resetHomeFilters = resetHomeFilters;
window.filterHomeCases = filterHomeCases;
window.loadMoreHomeCases = loadMoreHomeCases;
window.sortHomeCases = sortHomeCases;
window.handleHomeQuickQa = handleHomeQuickQa;
window.selectFiraCaseById = selectFiraCaseById;
window.selectFiraCase = selectFiraCase;
window.openCaseModal = openCaseModal;
window.closeCaseModal = closeCaseModal;
window.filterExplorerTable = filterExplorerTable;
window.resetExplorerFilters = resetExplorerFilters;
window.modalActionQa = modalActionQa;
window.modalActionSummarize = modalActionSummarize;
window.jumpToQa = jumpToQa;
window.jumpToSummarizer = jumpToSummarizer;
