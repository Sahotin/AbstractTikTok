(function () {
  const state = {
    context: null,
    latestContext: null,
    contextMode: "latest",
    submitting: false,
    historyFiles: [],
    selectedHistoryGroups: new Map(),
    historyNotice: ""
  };
  const platformNames = { douyin: "抖音", bilibili: "B站" };

  function createUi() {
    if (document.getElementById("analysis-launcher")) return;
    document.body.insertAdjacentHTML("beforeend", `
      <a id="analysis-results-link" class="analysis-results-link" href="/dashboard?latest=1">查看分析结果</a>
      <button id="analysis-launcher" class="analysis-launcher" type="button">
        <span class="dot"></span><span class="label">AI 智能分析</span>
      </button>
      <div id="analysis-modal-backdrop" class="analysis-modal-backdrop" aria-hidden="true">
        <section class="analysis-modal" role="dialog" aria-modal="true" aria-labelledby="analysis-modal-title">
          <div class="analysis-modal-header">
          <div><p class="analysis-eyebrow">CRAWL → INTELLIGENCE</p><h2 id="analysis-modal-title">分析本次采集数据</h2><p id="analysis-modal-description">选择需要的能力，系统会自动完成标准化、AI 分析和结果跳转。</p></div>
            <button id="analysis-close" class="analysis-close" type="button" aria-label="关闭">×</button>
          </div>
          <div id="analysis-context" class="analysis-context"><strong>正在读取采集结果…</strong></div>
          <div class="analysis-options">
            <label class="analysis-option"><input type="checkbox" value="sentiment" checked><span><strong>情感 / 情绪 / 立场</strong><small>DeepSeek 结构化识别观点倾向、情绪与关键词</small></span></label>
            <label class="analysis-option"><input type="checkbox" value="risk" checked><span><strong>风险检测</strong><small>识别高风险评论、风险等级及可解释原因</small></span></label>
            <label class="analysis-option"><input type="checkbox" value="trend" checked><span><strong>舆情趋势</strong><small>按时间展示评论量、情感与风险变化</small></span></label>
            <label class="analysis-option"><input type="checkbox" value="topics" checked><span><strong>主题与观点聚类</strong><small>Embedding 聚类、代表性评论与主要争议</small></span></label>
            <label class="analysis-option"><input type="checkbox" value="semantic_search" checked><span><strong>语义检索</strong><small>用自然语言搜索含义相近的评论</small></span></label>
            <label class="analysis-option"><input type="checkbox" value="agent" checked><span><strong>舆情 Agent</strong><small>DeepSeek 自主调用统计、趋势、主题与检索工具</small></span></label>
          </div>
          <div class="analysis-controls">
            <label>最多分析评论数<input id="analysis-limit" type="number" min="1" max="10000" value="100"></label>
            <label>并发请求数<input id="analysis-concurrency" type="number" min="1" max="20" value="2"></label>
          </div>
          <div class="analysis-cost-note">选择 DeepSeek 或 Embedding 能力会产生 API 调用费用。建议首次使用 100 条评论验证效果，再逐步扩大范围。</div>
          <div class="analysis-actions"><button id="analysis-cancel" class="analysis-cancel" type="button">取消</button><button id="analysis-submit" class="analysis-submit" type="button">开始分析并查看结果</button></div>
          <div id="analysis-error" class="analysis-error"></div>
        </section>
      </div>`);

    document.getElementById("analysis-launcher").addEventListener("click", openLatestModal);
    document.getElementById("analysis-close").addEventListener("click", closeModal);
    document.getElementById("analysis-cancel").addEventListener("click", closeModal);
    document.getElementById("analysis-submit").addEventListener("click", submitWorkflow);
    document.getElementById("analysis-modal-backdrop").addEventListener("click", function (event) {
      if (event.target === event.currentTarget) closeModal();
    });
    document.addEventListener("keydown", function (event) { if (event.key === "Escape") closeModal(); });
    enhanceHeaderNavigation();
  }

  function enhanceHeaderNavigation() {
    if (document.getElementById("analysis-header-link")) return;
    const banner = document.querySelector("header") || document.querySelector('[role="banner"]');
    if (!banner) return;
    const link = document.createElement("a");
    link.id = "analysis-header-link";
    link.className = "analysis-header-link";
    link.href = "/dashboard?latest=1";
    link.textContent = "智能分析";
    const apiText = Array.from(banner.querySelectorAll("span,div")).find(function (item) {
      return item.textContent && item.textContent.trim() === "API:";
    });
    if (apiText && apiText.parentElement) apiText.parentElement.insertBefore(link, apiText);
    else banner.appendChild(link);
  }

  async function api(url, options) {
    const response = await fetch(url, options);
    const payload = await response.json().catch(function () { return {}; });
    if (!response.ok) throw new Error(payload.detail || `请求失败 (${response.status})`);
    return payload;
  }

  async function refreshContext() {
    createUi();
    try {
      state.latestContext = await api("/api/crawler/analysis-context");
      if (state.contextMode === "latest") {
        state.context = state.latestContext;
        renderContext();
      } else {
        renderLauncher();
      }
    } catch (error) {
      state.latestContext = null;
      if (state.contextMode === "latest") {
        state.context = null;
        renderContext(error.message);
      } else {
        renderLauncher();
      }
    }
  }

  function renderLauncher() {
    const launcher = document.getElementById("analysis-launcher");
    if (!launcher) return;
    const ready = Boolean(state.latestContext && state.latestContext.available);
    launcher.classList.toggle("ready", ready);
    launcher.querySelector(".label").textContent = ready ? "采集完成 · AI 分析" : "AI 智能分析";
  }

  function renderContext(error) {
    const launcher = document.getElementById("analysis-launcher");
    const box = document.getElementById("analysis-context");
    const submit = document.getElementById("analysis-submit");
    const context = state.context;
    renderLauncher();
    submit.disabled = state.submitting || !(context && context.available);
    if (error) {
      box.innerHTML = "";
      const strong = document.createElement("strong"); strong.textContent = "无法读取采集结果";
      const small = document.createElement("small"); small.textContent = error;
      box.append(strong, small);
      return;
    }
    if (!context) return;
    box.innerHTML = "";
    const strong = document.createElement("strong");
    strong.textContent = context.available
      ? `${platformNames[context.analysis_platform] || context.analysis_platform} · ${context.files.length} 个${state.contextMode === "history" ? "历史" : "本次输出"}文件`
      : context.message;
    box.appendChild(strong);
    if (context.files.length) {
      const small = document.createElement("small");
      small.textContent = context.files.map(function (file) { return file.name; }).join("、");
      box.appendChild(small);
    }
  }

  function showModal() {
    document.getElementById("analysis-modal-backdrop").classList.add("open");
    document.getElementById("analysis-modal-backdrop").setAttribute("aria-hidden", "false");
  }

  async function openLatestModal() {
    state.contextMode = "latest";
    state.context = state.latestContext;
    document.getElementById("analysis-modal-title").textContent = "分析本次采集数据";
    document.getElementById("analysis-modal-description").textContent = "选择需要的能力，系统会自动完成标准化、AI 分析和结果跳转。";
    showModal();
    await refreshContext();
  }

  function openHistoryModal() {
    const groups = Array.from(state.selectedHistoryGroups.values());
    if (!groups.length) return;
    const filesByPath = new Map();
    groups.forEach(function (group) {
      group.files.forEach(function (file) { filesByPath.set(file.path, file); });
    });
    state.contextMode = "history";
    state.context = {
      available: true,
      supported: true,
      analysis_platform: groups[0].platform,
      crawler_type: groups.length === 1 ? groups[0].crawlerType : "import",
      files: Array.from(filesByPath.values()),
      message: "历史采集数据已就绪"
    };
    document.getElementById("analysis-modal-title").textContent = "分析已选择的历史数据";
    document.getElementById("analysis-modal-description").textContent = `已选择 ${groups.length} 条采集记录。请选择分析能力，完成后将自动进入分析页面。`;
    renderContext();
    showModal();
  }

  function closeModal() {
    if (state.submitting) return;
    document.getElementById("analysis-modal-backdrop").classList.remove("open");
    document.getElementById("analysis-modal-backdrop").setAttribute("aria-hidden", "true");
    state.contextMode = "latest";
    state.context = state.latestContext;
  }

  function buildHistoryGroups(files) {
    const groups = new Map();
    files.filter(function (file) { return file.analysis_supported; }).forEach(function (file) {
      const id = file.analysis_group_id;
      if (!groups.has(id)) groups.set(id, {
        id: id,
        platform: file.analysis_platform,
        crawlerType: file.crawler_type || "import",
        files: []
      });
      groups.get(id).files.push(file);
    });
    return groups;
  }

  function legacyGroupId(value) {
    let hash = 0;
    for (let index = 0; index < value.length; index += 1) {
      hash = ((hash << 5) - hash + value.charCodeAt(index)) | 0;
    }
    return `legacy-${Math.abs(hash)}`;
  }

  function normalizeHistoryFile(file) {
    if (typeof file.analysis_supported === "boolean" && file.analysis_group_id) return file;
    const path = String(file.path || "").replace(/\\/g, "/");
    const parts = path.toLowerCase().split("/");
    const aliases = { dy: "douyin", douyin: "douyin", bili: "bilibili", bilibili: "bilibili" };
    const platformPart = parts.find(function (part) { return aliases[part]; });
    const platform = platformPart ? aliases[platformPart] : null;
    const type = String(file.type || "").toLowerCase();
    const supported = Boolean(platform && ["json", "jsonl", "csv", "xlsx"].includes(type));
    const stem = String(file.name || "").replace(/\.[^.]+$/, "").toLowerCase();
    const tokens = stem.split("_");
    const crawlerType = tokens.find(function (token) { return ["search", "detail", "creator"].includes(token); }) || "import";
    const match = stem.match(/^(search|detail|creator)_(comments|contents|creators|videos|notes)_(.+)$/);
    const groupName = match && type !== "xlsx" ? `${match[1]}_${match[3]}` : String(file.name || "").toLowerCase();
    const parent = parts.slice(0, -1).join("/");
    return Object.assign({}, file, {
      analysis_supported: supported,
      analysis_platform: platform,
      analysis_group_id: legacyGroupId(`${platform || "unsupported"}/${parent}/${groupName}`),
      crawler_type: crawlerType
    });
  }

  function updateHistoryControls(dialog) {
    const toolbar = dialog.querySelector(".history-analysis-toolbar");
    if (!toolbar) return;
    const groups = Array.from(state.selectedHistoryGroups.values());
    const fileCount = groups.reduce(function (count, group) { return count + group.files.length; }, 0);
    toolbar.querySelector(".history-analysis-summary").textContent = state.historyNotice || (groups.length
      ? `已选择 ${groups.length} 条采集记录，共 ${fileCount} 个关联文件`
      : "勾选历史记录；同批内容与评论文件会自动关联");
    toolbar.querySelector("button").disabled = groups.length === 0;
    dialog.querySelectorAll(".history-analysis-check input").forEach(function (input) {
      input.checked = state.selectedHistoryGroups.has(input.dataset.groupId);
    });
  }

  function toggleHistoryGroup(group, checked, dialog) {
    if (!group) return;
    if (checked) {
      const selected = Array.from(state.selectedHistoryGroups.values());
      if (selected.length && selected[0].platform !== group.platform) {
        state.historyNotice = "一次分析只能选择同一平台的数据，请先取消当前选择";
        updateHistoryControls(dialog);
        return;
      }
      const selectedFileCount = selected.reduce(function (count, item) { return count + item.files.length; }, 0);
      if (!state.selectedHistoryGroups.has(group.id) && selectedFileCount + group.files.length > 20) {
        state.historyNotice = "一次最多提交 20 个关联文件，请减少选择数量";
        updateHistoryControls(dialog);
        return;
      }
      state.selectedHistoryGroups.set(group.id, group);
    } else {
      state.selectedHistoryGroups.delete(group.id);
    }
    state.historyNotice = "";
    updateHistoryControls(dialog);
  }

  function enhanceHistoryDialog() {
    const dialogs = Array.from(document.querySelectorAll('[role="dialog"]'));
    const dialog = dialogs.find(function (candidate) { return candidate.querySelector('h3[title]'); });
    if (!dialog || !state.historyFiles.length) return;
    const groups = buildHistoryGroups(state.historyFiles);
    if (!dialog.querySelector(".history-analysis-toolbar")) {
      const toolbar = document.createElement("div");
      toolbar.className = "history-analysis-toolbar";
      toolbar.innerHTML = '<div><strong>历史数据 AI 分析</strong><small class="history-analysis-summary">勾选历史记录；同批内容与评论文件会自动关联</small></div><button type="button" disabled>分析所选记录 →</button>';
      toolbar.querySelector("button").addEventListener("click", openHistoryModal);
      const explorer = dialog.querySelector('h3[title]').closest('.h-full.flex.flex-col');
      if (explorer) explorer.insertBefore(toolbar, explorer.children[1] || null);
    }

    const usedPaths = new Set();
    dialog.querySelectorAll('h3[title]').forEach(function (title) {
      const card = title.closest('[class*="card-scan"]');
      if (!card) return;
      if (card.dataset.analysisFilePath) {
        usedPaths.add(card.dataset.analysisFilePath);
        return;
      }
      const candidates = state.historyFiles.filter(function (file) {
        return file.name === title.getAttribute("title") && !usedPaths.has(file.path);
      });
      const file = candidates[0];
      if (!file) return;
      usedPaths.add(file.path);
      card.dataset.analysisFilePath = file.path;
      const label = document.createElement("label");
      label.className = "history-analysis-check";
      if (!file.analysis_supported) label.classList.add("unsupported");
      label.title = file.analysis_supported ? "选择这条采集记录进行分析" : "当前分析仅支持抖音和 B站的 JSON、JSONL、CSV、Excel 数据";
      label.innerHTML = `<input type="checkbox" data-group-id="${file.analysis_group_id}" ${file.analysis_supported ? "" : "disabled"}><span>${file.analysis_supported ? "选择分析" : "暂不支持"}</span>`;
      if (file.analysis_supported) {
        label.querySelector("input").addEventListener("change", function (event) {
          toggleHistoryGroup(groups.get(file.analysis_group_id), event.target.checked, dialog);
        });
      }
      card.appendChild(label);
    });
    updateHistoryControls(dialog);
  }

  async function refreshHistoryFiles() {
    try {
      const payload = await api("/api/data/files");
      state.historyFiles = (payload.files || []).map(normalizeHistoryFile);
      enhanceHistoryDialog();
    } catch (_) {
      state.historyFiles = [];
    }
  }

  async function submitWorkflow() {
    const errorBox = document.getElementById("analysis-error");
    errorBox.classList.remove("show");
    if (!state.context || !state.context.available) return refreshContext();
    const items = Array.from(document.querySelectorAll(".analysis-options input:checked")).map(function (input) { return input.value; });
    if (!items.length) {
      errorBox.textContent = "请至少选择一个分析项";
      errorBox.classList.add("show");
      return;
    }
    state.submitting = true;
    renderContext();
    document.getElementById("analysis-submit").textContent = "正在创建分析任务…";
    try {
      const workflow = await api("/api/analysis/workflows", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          platform: state.context.analysis_platform,
          source_files: state.context.files.map(function (file) { return file.path; }),
          analysis_items: items,
          crawler_type: state.context.crawler_type || "import",
          limit: Number(document.getElementById("analysis-limit").value || 100),
          batch_size: 20,
          max_concurrency: Number(document.getElementById("analysis-concurrency").value || 2)
        })
      });
      window.location.href = `/dashboard?workflow_id=${encodeURIComponent(workflow.id)}`;
    } catch (error) {
      state.submitting = false;
      document.getElementById("analysis-submit").textContent = "开始分析并查看结果";
      errorBox.textContent = error.message;
      errorBox.classList.add("show");
      renderContext();
    }
  }

  function initialize() {
    refreshContext();
    refreshHistoryFiles();
    const observer = new MutationObserver(function () {
      window.requestAnimationFrame(function () {
        enhanceHistoryDialog();
        enhanceHeaderNavigation();
      });
    });
    observer.observe(document.body, { childList: true, subtree: true });
    window.requestAnimationFrame(enhanceHeaderNavigation);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initialize);
  else initialize();
  window.setInterval(refreshContext, 3000);
})();
