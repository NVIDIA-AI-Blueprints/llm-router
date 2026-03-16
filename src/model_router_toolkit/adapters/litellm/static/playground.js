/* Model Router Toolkit — Playground UI */

const Playground = (function () {
  'use strict';

  // ── DOM refs ──────────────────────────────────────────────────────────
  const messagesEl   = document.getElementById('messages');
  const inputEl      = document.getElementById('input');
  const sendBtn      = document.getElementById('sendBtn');
  const tolSlider    = document.getElementById('tolSlider');
  const tolValue     = document.getElementById('tolValue');
  const welcomeEl    = document.getElementById('welcome');
  const howItWorksEl = document.getElementById('howItWorks');
  const headerSub    = document.getElementById('headerSubtitle');

  // ── State ─────────────────────────────────────────────────────────────
  let config       = {};
  let models       = [];
  let displayNames = {};
  let modelCosts   = {};
  let stats        = { queries: 0, routerCost: 0, maxCost: 0, modelCounts: {} };

  const MODEL_COLORS = [
    '#3b82f6', '#76B900', '#f59e0b', '#ef4444',
    '#8b5cf6', '#ec4899', '#14b8a6', '#f97316',
  ];

  function dn(name) { return displayNames[name] || name; }

  function modelColor(name) {
    const idx = models.findIndex(m => m.name === name);
    return MODEL_COLORS[idx % MODEL_COLORS.length];
  }

  function estimateTokens(text) { return Math.ceil(text.length / 4); }

  function scrollToBottom() { messagesEl.scrollTop = messagesEl.scrollHeight; }

  // ── Initialization ────────────────────────────────────────────────────
  async function init() {
    await Promise.all([loadConfig(), loadModels()]);
    setupInputHandlers();
    inputEl.focus();
  }

  async function loadConfig() {
    try {
      const resp = await fetch('/api/config');
      config = await resp.json();
    } catch (e) {
      config = { routing_method: 'unknown', review_available: false };
    }

    const method = (config.routing_method || 'unknown').toLowerCase();
    headerSub.textContent = method.charAt(0).toUpperCase() + method.slice(1) + ' Router \u2022 Playground';

    if (config.review_available) {
      document.getElementById('reviewSection').style.display = '';
      const judgeLabel = document.getElementById('reviewJudgeLabel');
      if (config.judge_model) {
        judgeLabel.textContent = config.judge_model + ' as judge';
      }
    }

    if (config.device === 'cpu') {
      document.getElementById('cpuBanner').style.display = '';
    }

    updateHowItWorks(method);
  }

  function updateHowItWorks(method) {
    if (!howItWorksEl) return;
    if (method === 'prefill') {
      howItWorksEl.innerHTML =
        '<strong>How it works:</strong>' +
        '<ol>' +
        '<li>Your question is run through an encoder model (single forward pass)</li>' +
        '<li>Hidden state features are extracted and transformed</li>' +
        '<li>An MLP ensemble scores p(correct) for each model</li>' +
        '<li>The router picks the <strong>most efficient</strong> model above the accuracy threshold</li>' +
        '</ol>';
    } else {
      howItWorksEl.innerHTML =
        '<strong>How it works:</strong>' +
        '<ol>' +
        '<li>Your question is analyzed by the routing model</li>' +
        '<li>Each model receives a p(correct) confidence score</li>' +
        '<li>The router picks the <strong>most efficient</strong> model above the accuracy threshold</li>' +
        '</ol>';
    }
  }

  async function loadModels() {
    try {
      const resp = await fetch('/api/models');
      models = await resp.json();
    } catch (e) {
      models = [];
    }

    models.forEach(function (m) {
      displayNames[m.name] = m.display_name || m.name;
      modelCosts[m.name] = {
        input: m.cost_per_m_input_tokens || 0,
        output: m.cost_per_m_output_tokens || 0,
      };
    });
    renderModelToggles();
  }

  // ── Model Toggles ────────────────────────────────────────────────────
  function renderModelToggles() {
    var container = document.getElementById('modelToggles');
    container.innerHTML = models.map(function (m) {
      var outCost = (m.cost_per_m_output_tokens || 0).toFixed(2);
      return '<div class="model-toggle" data-model="' + m.name + '">' +
        '<label class="toggle-switch">' +
        '<input type="checkbox" checked data-model="' + m.name + '">' +
        '<span class="toggle-slider"></span></label>' +
        '<div class="model-toggle-info">' +
        '<div class="model-toggle-name">' + (m.display_name || m.name) + '</div>' +
        '<div class="model-toggle-cost">$' + outCost + '/M output</div>' +
        '</div></div>';
    }).join('');

    container.querySelectorAll('input[type="checkbox"]').forEach(function (cb) {
      cb.addEventListener('change', function () {
        var checked = container.querySelectorAll('input:checked');
        if (checked.length === 0) { cb.checked = true; return; }
        cb.closest('.model-toggle').classList.toggle('disabled', !cb.checked);
      });
    });
  }

  function getEnabledModels() {
    return Array.from(document.querySelectorAll('#modelToggles input:checked'))
      .map(function (cb) { return cb.dataset.model; });
  }

  // ── Input Handlers ────────────────────────────────────────────────────
  function setupInputHandlers() {
    inputEl.addEventListener('input', function () {
      inputEl.style.height = 'auto';
      inputEl.style.height = Math.min(inputEl.scrollHeight, 100) + 'px';
    });

    inputEl.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    });

    sendBtn.addEventListener('click', sendMessage);

    tolSlider.addEventListener('input', function () {
      tolValue.textContent = parseFloat(tolSlider.value).toFixed(2);
    });
  }

  function usePrompt(text) {
    inputEl.value = text;
    sendMessage();
  }

  // ── Cost Helpers ──────────────────────────────────────────────────────
  function costForTokens(modelName, inputTok, outputTok) {
    var c = modelCosts[modelName];
    if (!c) return 0;
    return (inputTok * c.input + outputTok * c.output) / 1_000_000;
  }

  function mostExpensiveModel() {
    var best = null;
    var bestCost = -1;
    models.forEach(function (m) {
      var c = (m.cost_per_m_output_tokens || 0);
      if (c > bestCost) { bestCost = c; best = m.name; }
    });
    return best;
  }

  // ── Pipeline Visualization ────────────────────────────────────────────
  function renderPipelineVis(routingData, ttftMs) {
    var meta = routingData.metadata || {};
    var method = (config.routing_method || 'router').toUpperCase();
    var routeMs = routingData.route_ms || 0;
    var routeLabel = routeMs < 1 ? routeMs.toFixed(1) : Math.round(routeMs);

    var steps = [];

    steps.push({ label: method, value: routeLabel + 'ms', detail: 'route decision', active: true });

    var thresholdVal = meta.threshold != null ? meta.threshold.toFixed(3) : '--';
    steps.push({ label: 'Threshold', value: '\u2265 ' + thresholdVal, detail: 'p_max ' + (meta.p_max != null ? meta.p_max.toFixed(3) : '--'), active: true });

    var selCost = modelCosts[routingData.selected_model];
    var costLabel = selCost ? '$' + selCost.output.toFixed(2) + '/M' : '';
    steps.push({ label: 'Selected', value: dn(routingData.selected_model), detail: costLabel, active: true });

    var ttftActive = ttftMs != null;
    steps.push({ label: 'TTFT', value: ttftActive ? ttftMs + 'ms' : '\u2026', detail: 'first token', active: ttftActive, isTtft: true });

    var html = '<div class="pipeline-vis">';
    steps.forEach(function (s, i) {
      if (i > 0) {
        var arrowActive = s.active && steps[i - 1].active;
        html += '<div class="pipeline-arrow' + (arrowActive ? ' active' : '') + '">\u203A</div>';
      }
      var cls = 'pipeline-step' + (s.active ? ' active' : ' pending');
      if (s.isTtft) cls += ' pipeline-ttft-step';
      html += '<div class="' + cls + '">';
      html += '<div class="pipeline-step-label">' + s.label + '</div>';
      html += '<div class="pipeline-step-value' + (s.isTtft ? ' pipeline-ttft-value' : '') + '">' + s.value + '</div>';
      html += '<div class="pipeline-step-detail">' + s.detail + '</div>';
      html += '</div>';
    });
    html += '</div>';

    html += '<div class="pipeline-timing-summary">';
    html += '<span class="timing-chip route">Route ' + routeLabel + 'ms</span>';
    html += '<span class="timing-chip ttft pipeline-ttft-chip">' + (ttftActive ? 'TTFT ' + ttftMs + 'ms' : 'TTFT \u2026') + '</span>';
    html += '</div>';

    return html;
  }

  function updateTtftDisplay(container, ttftMs) {
    var step = container.querySelector('.pipeline-ttft-step');
    var val = container.querySelector('.pipeline-ttft-value');
    var chip = container.querySelector('.pipeline-ttft-chip');
    if (step) { step.classList.remove('pending'); step.classList.add('active'); }
    if (val) val.textContent = ttftMs + 'ms';
    if (chip) chip.textContent = 'TTFT ' + ttftMs + 'ms';
  }

  // ── Routing Card ──────────────────────────────────────────────────────
  function renderRoutingCard(routingData, ttftMs) {
    var selected = routingData.selected_model;
    var modelNames = routingData.model_names || [];
    var confidences = routingData.confidences || [];
    var meta = routingData.metadata || {};
    var enabledSet = new Set(getEnabledModels());

    var probs = {};
    modelNames.forEach(function (name, i) { probs[name] = confidences[i]; });

    var bestModel = modelNames[0] || '';
    var bestProb = 0;
    modelNames.forEach(function (name, i) {
      if (confidences[i] > bestProb) { bestProb = confidences[i]; bestModel = name; }
    });

    var selCost = modelCosts[selected];
    var costStr = selCost ? '$' + selCost.output.toFixed(2) + '/M' : '';
    var routeMs = routingData.route_ms ? Math.round(routingData.route_ms) + 'ms' : '';

    var cardId = 'rc-' + Date.now();
    var probsId = 'probs-' + Date.now();
    var pipelineHtml = renderPipelineVis(routingData, ttftMs);

    var probsHtml = '';
    modelNames.forEach(function (name, i) {
      var prob = confidences[i];
      var pct = (prob * 100).toFixed(0);
      var isBest = name === bestModel;
      var isSelected = name === selected;
      var enabled = enabledSet.has(name);
      var marker = '';
      if (isBest && isSelected) marker = '<span class="prob-marker selected">\u2605\u2190</span>';
      else if (isBest) marker = '<span class="prob-marker best">\u2605</span>';
      else if (isSelected) marker = '<span class="prob-marker selected">\u2190$</span>';
      else marker = '<span class="prob-marker"></span>';

      probsHtml += '<div class="prob-row" style="' + (enabled ? '' : 'opacity:0.35') + '">' +
        '<span class="prob-name">' + dn(name) + '</span>' +
        '<span class="prob-value">' + prob.toFixed(3) + '</span>' +
        '<div class="prob-bar-track"><div class="prob-bar-fill" style="width:' + pct + '%"></div></div>' +
        marker + '</div>';
    });

    var thresholdStr = meta.threshold != null ? meta.threshold.toFixed(3) : '--';

    return '<div class="routing-card">' +
      '<div class="routing-header">' +
      '<div class="routing-badge">' +
      '<span class="dot"></span>' +
      '<span class="model-name">' + dn(selected) + '</span>' +
      '<span class="cost">' + costStr + '</span>' +
      '<span class="latency">' + routeMs + '</span>' +
      '</div>' +
      '<button class="routing-collapse-btn" onclick="var d=document.getElementById(\'' + cardId + '\');var c=d.classList.toggle(\'collapsed\');this.textContent=c?\'Show details\':\'Hide details\'">Hide details</button>' +
      '</div>' +
      '<div class="routing-details" id="' + cardId + '">' +
      pipelineHtml +
      '<div style="margin-top:8px;">' +
      '<div style="margin-bottom:4px; font-size:10px; color:var(--text-dim);">\u2605 = highest probability, \u2190 = selected model</div>' +
      probsHtml +
      '<div class="routing-summary"><strong>Selected: ' + dn(selected) + '</strong> \u2014 most efficient model with p(correct) \u2265 ' + thresholdStr + '</div>' +
      '</div></div></div>';
  }

  // ── Send Message ──────────────────────────────────────────────────────
  async function sendMessage() {
    var text = inputEl.value.trim();
    if (!text) return;

    if (welcomeEl && welcomeEl.parentNode) welcomeEl.remove();
    var promptExEl = document.getElementById('promptExamples');
    if (promptExEl) promptExEl.remove();

    var userDiv = document.createElement('div');
    userDiv.className = 'msg user';
    userDiv.textContent = text;
    messagesEl.appendChild(userDiv);

    inputEl.value = '';
    inputEl.style.height = 'auto';
    sendBtn.disabled = true;
    inputEl.disabled = true;

    var typingDiv = document.createElement('div');
    typingDiv.className = 'typing';
    typingDiv.innerHTML = '<span></span><span></span><span></span>';
    messagesEl.appendChild(typingDiv);
    scrollToBottom();

    var payload = {
      message: text,
      tolerance: parseFloat(tolSlider.value),
      enabled_models: getEnabledModels(),
    };

    var routingData = null;
    var reasoningText = '';
    var contentText = '';
    var assistantDiv = null;
    var reasoningEl = null;
    var contentEl = null;
    var llmStartTime = null;
    var ttftMs = null;
    var ttftRecorded = false;

    try {
      var resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      typingDiv.remove();

      assistantDiv = document.createElement('div');
      assistantDiv.className = 'msg assistant';
      messagesEl.appendChild(assistantDiv);

      var reader = resp.body.getReader();
      var decoder = new TextDecoder();
      var buffer = '';
      var eventType = null;

      while (true) {
        var chunk = await reader.read();
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, { stream: true });

        var lines = buffer.split('\n');
        buffer = lines.pop();

        for (var li = 0; li < lines.length; li++) {
          var line = lines[li];
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim();
          } else if (line.startsWith('data: ') && eventType) {
            try {
              var data = JSON.parse(line.slice(6));
              handleEvent(eventType, data);
            } catch (e) { /* skip unparseable */ }
            eventType = null;
          }
        }
        scrollToBottom();
      }
    } catch (e) {
      if (typingDiv.parentNode) typingDiv.remove();
      var errDiv = document.createElement('div');
      errDiv.className = 'msg error';
      errDiv.textContent = 'Connection error: ' + e.message;
      messagesEl.appendChild(errDiv);
    } finally {
      if (typingDiv.parentNode) typingDiv.remove();
      sendBtn.disabled = false;
      inputEl.disabled = false;
      inputEl.focus();
    }

    function handleEvent(type, data) {
      if (type === 'routing') {
        routingData = data;
        llmStartTime = Date.now();
        assistantDiv.innerHTML = renderRoutingCard(data);
        stats.queries++;
        stats.modelCounts[data.selected_model] = (stats.modelCounts[data.selected_model] || 0) + 1;
        updateStats();

      } else if (type === 'reasoning') {
        recordTtft();
        if (!reasoningEl) {
          var block = document.createElement('div');
          block.className = 'reasoning-block';
          block.innerHTML = '<div class="reasoning-toggle" onclick="this.nextElementSibling.classList.toggle(\'open\')">\u25B8 Show reasoning</div><div class="reasoning-content"></div>';
          assistantDiv.appendChild(block);
          reasoningEl = block.querySelector('.reasoning-content');
        }
        reasoningText += data.text;
        reasoningEl.textContent = reasoningText;

      } else if (type === 'token') {
        recordTtft();
        if (!contentEl) {
          contentEl = document.createElement('div');
          contentEl.className = 'response-content';
          assistantDiv.appendChild(contentEl);
        }
        contentText += data.text;
        contentEl.innerHTML = marked.parse(contentText);

      } else if (type === 'done') {
        var finalText = contentText || reasoningText;

        if (finalText && routingData) {
          var inputTok = estimateTokens(text);
          var outputTok = estimateTokens(finalText);
          var selCost = costForTokens(routingData.selected_model, inputTok, outputTok);
          var maxModel = mostExpensiveModel();
          var maxCost = maxModel ? costForTokens(maxModel, inputTok, outputTok) : selCost;
          stats.routerCost += selCost;
          stats.maxCost += maxCost;
          updateStats();
        }

        if (finalText) {
          appendCopyButton(assistantDiv, finalText);
        }

        if (document.getElementById('autoReviewToggle').checked && finalText && routingData) {
          startReview(text, finalText, routingData.selected_model, getEnabledModels(), assistantDiv);
        }

      } else if (type === 'error') {
        var errEl = document.createElement('div');
        errEl.className = 'msg error';
        errEl.textContent = data.message;
        messagesEl.appendChild(errEl);
      }
    }

    function recordTtft() {
      if (!ttftRecorded && llmStartTime) {
        ttftRecorded = true;
        ttftMs = Date.now() - llmStartTime;
        if (assistantDiv) updateTtftDisplay(assistantDiv, ttftMs);
      }
    }

    scrollToBottom();
  }

  // ── Copy Button ───────────────────────────────────────────────────────
  function appendCopyButton(container, text) {
    var btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg> Copy';
    btn.onclick = function () {
      navigator.clipboard.writeText(text);
      btn.classList.add('copied');
      btn.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg> Copied';
      setTimeout(function () {
        btn.classList.remove('copied');
        btn.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg> Copy';
      }, 2000);
    };
    container.appendChild(btn);
  }

  // ── Session Stats ─────────────────────────────────────────────────────
  function updateStats() {
    var statsEl = document.getElementById('statsSection');
    if (statsEl && stats.queries === 1) statsEl.classList.remove('collapsed');

    document.getElementById('statQueries').textContent = stats.queries;

    var hasCost = stats.maxCost > 0;
    document.getElementById('statRouterCost').textContent = hasCost
      ? '$' + stats.routerCost.toFixed(6)
      : '--';
    document.getElementById('statMaxCost').textContent = hasCost
      ? '$' + stats.maxCost.toFixed(6)
      : '--';

    if (hasCost && stats.maxCost > 0) {
      var pct = ((1 - stats.routerCost / stats.maxCost) * 100).toFixed(0);
      var saved = stats.maxCost - stats.routerCost;
      document.getElementById('statSavings').textContent = pct + '% ($' + saved.toFixed(6) + ' saved)';
    } else {
      document.getElementById('statSavings').textContent = '--';
    }

    var usageEl = document.getElementById('modelUsage');
    var counts = stats.modelCounts;
    var keys = Object.keys(counts);
    if (keys.length === 0) {
      usageEl.innerHTML = '<div style="font-size:11px;color:var(--text-dim)">No queries yet</div>';
      return;
    }

    keys.sort(function (a, b) { return counts[b] - counts[a]; });
    usageEl.innerHTML = keys.map(function (name) {
      var count = counts[name];
      var pct = ((count / stats.queries) * 100).toFixed(0);
      return '<div class="model-usage-item">' +
        '<span class="model-usage-dot" style="background:' + modelColor(name) + '"></span>' +
        '<span class="model-usage-name">' + dn(name) + '</span>' +
        '<span class="model-usage-count">' + count + ' (' + pct + '%)</span>' +
        '</div>';
    }).join('');
  }

  // ── Auto Review ───────────────────────────────────────────────────────
  async function startReview(question, answer, selectedModel, enabledModels, targetDiv) {
    var reviewContainer = document.createElement('div');
    reviewContainer.className = 'review-container';
    targetDiv.appendChild(reviewContainer);

    var payload = {
      question: question,
      answer: answer,
      selected_model: selectedModel,
      enabled_models: enabledModels,
    };

    var comparisonDiv = null;

    try {
      var resp = await fetch('/api/review', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      var reader = resp.body.getReader();
      var decoder = new TextDecoder();
      var buffer = '';
      var eventType = null;

      while (true) {
        var chunk = await reader.read();
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, { stream: true });

        var lines = buffer.split('\n');
        buffer = lines.pop();

        for (var li = 0; li < lines.length; li++) {
          var line = lines[li];
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim();
          } else if (line.startsWith('data: ') && eventType) {
            try {
              var data = JSON.parse(line.slice(6));
              handleReviewEvent(eventType, data);
            } catch (e) { /* skip */ }
            eventType = null;
          }
        }
        scrollToBottom();
      }
    } catch (e) {
      reviewContainer.innerHTML = '<div class="review-card error">Review failed: ' + e.message + '</div>';
    }

    function handleReviewEvent(type, data) {
      if (type === 'judging') {
        reviewContainer.innerHTML = '<div class="review-card judging review-pulse">' +
          '<span class="verdict-icon">\uD83D\uDD0D</span> ' + data.status + '</div>';

      } else if (type === 'verdict') {
        var isCorrect = data.correct === true;
        var icon = isCorrect ? '\u2713' : '\u2717';
        var cls = isCorrect ? 'correct' : 'incorrect';
        var conf = data.confidence || 'medium';
        reviewContainer.innerHTML = '<div class="review-card ' + cls + '">' +
          '<span class="verdict-icon">' + icon + '</span> ' +
          '<strong>' + (isCorrect ? 'Likely correct' : 'May be incorrect') + '</strong> (' + conf + ' confidence)' +
          '<div style="margin-top:3px; font-size:10px;">' + (data.explanation || '') + '</div>' +
          '</div>';

      } else if (type === 'comparing') {
        comparisonDiv = document.createElement('div');
        comparisonDiv.className = 'review-card comparing review-pulse';
        comparisonDiv.innerHTML = '<span class="verdict-icon">\uD83D\uDD04</span> ' + data.status;
        reviewContainer.appendChild(comparisonDiv);

      } else if (type === 'model-result') {
        if (comparisonDiv) comparisonDiv.classList.remove('review-pulse');
        if (!comparisonDiv) {
          comparisonDiv = document.createElement('div');
          comparisonDiv.className = 'review-card comparing';
          reviewContainer.appendChild(comparisonDiv);
        }
        var mIcon = data.correct === true ? '\u2713' : data.correct === false ? '\u2717' : '?';
        var mColor = data.correct === true ? 'var(--green)' : data.correct === false ? '#f59e0b' : 'var(--text-dim)';
        var row = document.createElement('div');
        row.className = 'review-model-row';
        row.innerHTML =
          '<div class="review-model-icon" style="color:' + mColor + '">' + mIcon + '</div>' +
          '<div class="review-model-info">' +
          '<div class="review-model-name">' + (data.display_name || data.model) + '</div>' +
          '<div class="review-model-detail">' + (data.explanation || '') + '</div>' +
          '</div>';
        comparisonDiv.appendChild(row);

      } else if (type === 'comparison-done') {
        if (comparisonDiv) {
          comparisonDiv.classList.remove('review-pulse', 'comparing');
          comparisonDiv.classList.add(data.any_correct ? 'correct' : 'incorrect');
          var summaryDiv = document.createElement('div');
          summaryDiv.className = 'review-summary';
          summaryDiv.innerHTML = '<strong>' + data.summary + '</strong>';
          comparisonDiv.appendChild(summaryDiv);
        }
      }
    }
  }

  // ── Collapsible Sections ──────────────────────────────────────────────
  function toggleSection(headerEl) {
    var section = headerEl.closest('.sidebar-section');
    if (section) section.classList.toggle('collapsed');
  }

  // ── Mobile Sidebar Toggle ──────────────────────────────────────────────
  (function initSidebarToggle() {
    var btn = document.getElementById('sidebarToggle');
    var backdrop = document.getElementById('sidebarBackdrop');
    var sidebar = document.querySelector('.sidebar');
    if (!btn || !sidebar) return;

    function openSidebar()  { sidebar.classList.add('open'); backdrop.classList.add('open'); btn.classList.add('open'); }
    function closeSidebar() { sidebar.classList.remove('open'); backdrop.classList.remove('open'); btn.classList.remove('open'); }

    btn.addEventListener('click', function () {
      sidebar.classList.contains('open') ? closeSidebar() : openSidebar();
    });
    backdrop.addEventListener('click', closeSidebar);
  })();

  // ── Public API ────────────────────────────────────────────────────────
  init();

  return {
    usePrompt: usePrompt,
    toggleSection: toggleSection,
  };
})();
