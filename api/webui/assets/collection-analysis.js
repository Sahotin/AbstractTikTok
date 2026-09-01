(function () {
  const state = { context: null, submitting: false };
  const platformNames = { douyin: "抖音", bilibili: "B站" };

  function createUi() {
    if (document.getElementById("analysis-launcher")) return;
    document.body.insertAdjacentHTML("beforeend", `
      <button id="analysis-launcher" class="analysis-launcher" type="button">
        <span class="dot"></span><span class="label">AI 智能分析</span>
      </button>
      <div id="analysis-modal-backdrop" class="analysis-modal-backdrop" aria-hidden="true">
        <section class="analysis-modal" role="dialog" aria-modal="true" aria-labelledby="analysis-modal-title">
          <div class="analysis-modal-header">
            <div><p class="analysis-eyebrow">CRAWL → INTELLIGENCE</p><h2 id="analysis-modal-title">分析本次采集数据</h2><p>选择需要的能力，系统会自动完成标准化、AI 分析和结果跳转。</p></div>
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

    document.getElementById("analysis-launcher").addEventListener("click", openModal);
    document.getElementById("analysis-close").addEventListener("click", closeModal);
    document.getElementById("analysis-cancel").addEventListener("click", closeModal);
    document.getElementById("analysis-submit").addEventListener("click", submitWorkflow);
    document.getElementById("analysis-modal-backdrop").addEventListener("click", function (event) {
      if (event.target === event.currentTarget) closeModal();
    });
    document.addEventListener("keydown", function (event) { if (event.key === "Escape") closeModal(); });
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
      state.context = await api("/api/crawler/analysis-context");
      renderContext();
    } catch (error) {
      state.context = null;
      renderContext(error.message);
    }
  }

  function renderContext(error) {
    const launcher = document.getElementById("analysis-launcher");
    const label = launcher.querySelector(".label");
    const box = document.getElementById("analysis-context");
    const submit = document.getElementById("analysis-submit");
    const context = state.context;
    launcher.classList.toggle("ready", Boolean(context && context.available));
    label.textContent = context && context.available ? "采集完成 · AI 分析" : "AI 智能分析";
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
      ? `${platformNames[context.analysis_platform] || context.analysis_platform} · ${context.files.length} 个本次输出文件`
      : context.message;
    box.appendChild(strong);
    if (context.files.length) {
      const small = document.createElement("small");
      small.textContent = context.files.map(function (file) { return file.name; }).join("、");
      box.appendChild(small);
    }
  }

  async function openModal() {
    document.getElementById("analysis-modal-backdrop").classList.add("open");
    document.getElementById("analysis-modal-backdrop").setAttribute("aria-hidden", "false");
    await refreshContext();
  }

  function closeModal() {
    if (state.submitting) return;
    document.getElementById("analysis-modal-backdrop").classList.remove("open");
    document.getElementById("analysis-modal-backdrop").setAttribute("aria-hidden", "true");
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

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", refreshContext);
  else refreshContext();
  window.setInterval(refreshContext, 3000);
})();
