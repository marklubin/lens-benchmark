from __future__ import annotations


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def generate_human_ui_html() -> str:
    """Generate the self-contained HTML/CSS/JS page for the human benchmark."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LENS Human Benchmark</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 960px; margin: 0 auto; padding: 1rem;
    color: #1e293b; background: #f8fafc;
  }
  h1 { font-size: 1.5rem; margin-bottom: .25rem; }
  .subtitle { color: #64748b; font-size: .85rem; margin-bottom: 1rem; }

  /* Progress bar */
  .progress-bar-container {
    background: #e2e8f0; border-radius: .5rem; height: 1.2rem;
    margin-bottom: .5rem; overflow: hidden; position: relative;
  }
  .progress-bar-fill {
    height: 100%; background: #3b82f6; border-radius: .5rem;
    transition: width .3s;
  }
  .progress-label {
    font-size: .8rem; color: #64748b; margin-bottom: 1rem;
  }

  /* Episode feed */
  .episode-feed {
    max-height: 50vh; overflow-y: auto; border: 1px solid #e2e8f0;
    border-radius: .5rem; padding: .75rem; margin-bottom: 1rem;
    background: #fff;
  }
  .episode-card {
    border: 1px solid #e2e8f0; border-radius: .4rem;
    padding: .6rem .8rem; margin-bottom: .5rem;
    background: #f8fafc;
  }
  .episode-card:last-child { margin-bottom: 0; }
  .episode-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: .3rem; font-size: .78rem; color: #64748b;
  }
  .episode-header label {
    display: flex; align-items: center; gap: .3rem; cursor: pointer;
  }
  .episode-header input[type="checkbox"] { cursor: pointer; }
  .episode-text { font-size: .88rem; line-height: 1.5; white-space: pre-wrap; }
  .episode-id { font-weight: 600; color: #475569; }

  /* Buttons */
  .btn {
    display: inline-block; padding: .5rem 1.2rem;
    border: none; border-radius: .4rem; font-size: .9rem;
    cursor: pointer; font-weight: 500; transition: background .15s;
  }
  .btn-primary { background: #3b82f6; color: #fff; }
  .btn-primary:hover { background: #2563eb; }
  .btn-primary:disabled { background: #93c5fd; cursor: not-allowed; }
  .btn-success { background: #22c55e; color: #fff; }
  .btn-success:hover { background: #16a34a; }

  /* Question panel */
  .question-panel {
    border: 2px solid #3b82f6; border-radius: .5rem;
    padding: 1rem; margin-bottom: 1rem; background: #eff6ff;
    display: none;
  }
  .question-panel.active { display: block; }
  .question-prompt {
    font-size: .95rem; font-weight: 600; margin-bottom: .75rem;
    line-height: 1.4;
  }
  .question-meta {
    font-size: .78rem; color: #64748b; margin-bottom: .5rem;
  }
  textarea {
    width: 100%; min-height: 120px; padding: .6rem;
    border: 1px solid #cbd5e1; border-radius: .4rem;
    font-family: inherit; font-size: .88rem; line-height: 1.5;
    resize: vertical; margin-bottom: .5rem;
  }
  .cite-hint {
    font-size: .78rem; color: #64748b; margin-bottom: .75rem;
  }
  .timer {
    font-size: .8rem; color: #64748b; margin-bottom: .5rem;
  }
  .validation-msg {
    font-size: .8rem; color: #ef4444; margin-bottom: .5rem;
    display: none;
  }

  /* Completion */
  .completion-panel {
    text-align: center; padding: 2rem;
    border: 2px solid #22c55e; border-radius: .5rem;
    background: #f0fdf4; display: none;
  }
  .completion-panel.active { display: block; }
  .completion-panel h2 { font-size: 1.3rem; margin-bottom: .75rem; color: #16a34a; }
  .completion-panel p { margin-bottom: .75rem; color: #475569; font-size: .9rem; }
  .output-path {
    font-family: monospace; background: #1e293b; color: #e2e8f0;
    padding: .5rem .8rem; border-radius: .3rem; display: inline-block;
    margin-bottom: .75rem; font-size: .85rem;
  }
  .instructions { text-align: left; margin-top: 1rem; }
  .instructions code {
    background: #e2e8f0; padding: .1rem .3rem; border-radius: .2rem;
    font-size: .82rem;
  }

  .action-bar {
    display: flex; gap: .5rem; align-items: center;
    margin-bottom: 1rem;
  }

  /* Loading overlay */
  .loading { opacity: .6; pointer-events: none; }
</style>
</head>
<body>

<h1>LENS Human Benchmark</h1>
<div class="subtitle" id="run-info">Loading...</div>

<div class="progress-bar-container">
  <div class="progress-bar-fill" id="progress-fill" style="width:0%"></div>
</div>
<div class="progress-label" id="progress-label">--</div>

<div class="episode-feed" id="episode-feed">
  <div style="color:#94a3b8;text-align:center;padding:2rem;">
    Click "Next Episode" to begin reading.
  </div>
</div>

<div class="action-bar" id="action-bar">
  <button class="btn btn-primary" id="next-btn" onclick="nextEpisode()">Next Episode</button>
</div>

<div class="question-panel" id="question-panel">
  <div class="question-meta" id="question-meta"></div>
  <div class="question-prompt" id="question-prompt"></div>
  <div class="cite-hint">Check the episode boxes above to cite evidence for your answer.</div>
  <textarea id="answer-text" placeholder="Type your answer here..."></textarea>
  <div class="timer" id="timer">Time: 0s</div>
  <div class="validation-msg" id="validation-msg">Please provide an answer and cite at least one episode.</div>
  <button class="btn btn-primary" id="submit-btn" onclick="submitAnswer()">Submit Answer</button>
</div>

<div class="completion-panel" id="completion-panel">
  <h2>Benchmark Complete!</h2>
  <p>You have answered all questions across all scopes.</p>
  <button class="btn btn-success" id="finish-btn" onclick="finishBenchmark()">Finish &amp; Save Results</button>
  <div id="finish-result"></div>
</div>

<script>
(function() {
  "use strict";

  // State
  let state = null;
  let allEpisodes = [];
  let currentQuestions = [];
  let currentQuestionIndex = 0;
  let timerStart = null;
  let timerInterval = null;
  let answering = false;

  async function api(method, path, body) {
    const opts = { method, headers: { "Content-Type": "application/json" } };
    if (body) opts.body = JSON.stringify(body);
    const resp = await fetch(path, opts);
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(text || resp.statusText);
    }
    return resp.json();
  }

  async function loadState() {
    state = await api("GET", "/api/state");
    renderProgress();
  }

  async function loadEpisodes() {
    const data = await api("GET", "/api/episodes");
    allEpisodes = data.episodes || [];
    renderEpisodeFeed();
  }

  function renderProgress() {
    if (!state) return;
    const info = document.getElementById("run-info");
    info.textContent = "Run: " + state.run_id + " | Dataset: " + state.dataset_version;

    const scopeId = state.current_scope_id;
    const label = document.getElementById("progress-label");
    const fill = document.getElementById("progress-fill");

    if (state.is_complete) {
      label.textContent = "All scopes complete!";
      fill.style.width = "100%";
      document.getElementById("action-bar").style.display = "none";
      document.getElementById("question-panel").classList.remove("active");
      document.getElementById("completion-panel").classList.add("active");
      return;
    }

    if (!scopeId) return;
    const sp = state.scope_progress[scopeId];
    const scopeIdx = state.current_scope_index + 1;
    const totalScopes = state.scope_order.length;
    const pct = sp.total_episodes > 0 ? (sp.episodes_revealed / sp.total_episodes) * 100 : 0;
    fill.style.width = pct.toFixed(1) + "%";
    label.textContent = "Episode " + sp.episodes_revealed + "/" + sp.total_episodes
      + " — Scope: " + scopeId + " (" + scopeIdx + "/" + totalScopes + ")";
  }

  function renderEpisodeFeed() {
    const feed = document.getElementById("episode-feed");
    if (allEpisodes.length === 0) {
      feed.innerHTML = '<div style="color:#94a3b8;text-align:center;padding:2rem;">Click "Next Episode" to begin reading.</div>';
      return;
    }
    feed.innerHTML = allEpisodes.map(function(ep) {
      return '<div class="episode-card" data-id="' + esc(ep.episode_id) + '">'
        + '<div class="episode-header">'
        + '<span class="episode-id">' + esc(ep.episode_id) + '</span>'
        + '<span>' + esc(ep.timestamp) + '</span>'
        + '<label><input type="checkbox" class="ref-check" value="' + esc(ep.episode_id) + '"'
        + (answering ? '' : ' disabled') + '> cite</label>'
        + '</div>'
        + '<div class="episode-text">' + esc(ep.text) + '</div>'
        + '</div>';
    }).join("");
    feed.scrollTop = feed.scrollHeight;
  }

  function esc(s) {
    var d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function setCheckboxes(enabled) {
    var checks = document.querySelectorAll(".ref-check");
    for (var i = 0; i < checks.length; i++) {
      checks[i].disabled = !enabled;
      if (!enabled) checks[i].checked = false;
    }
  }

  function getCheckedRefs() {
    var refs = [];
    var checks = document.querySelectorAll(".ref-check:checked");
    for (var i = 0; i < checks.length; i++) refs.push(checks[i].value);
    return refs;
  }

  function startTimer() {
    timerStart = Date.now();
    var el = document.getElementById("timer");
    if (timerInterval) clearInterval(timerInterval);
    timerInterval = setInterval(function() {
      var s = Math.floor((Date.now() - timerStart) / 1000);
      el.textContent = "Time: " + s + "s";
    }, 1000);
  }

  function stopTimer() {
    if (timerInterval) clearInterval(timerInterval);
    timerInterval = null;
    return timerStart ? Date.now() - timerStart : 0;
  }

  function showQuestion(q) {
    answering = true;
    setCheckboxes(true);
    var panel = document.getElementById("question-panel");
    panel.classList.add("active");
    document.getElementById("question-meta").textContent =
      "Question " + (currentQuestionIndex + 1) + "/" + currentQuestions.length
      + " | Type: " + q.question_type + " | Checkpoint: " + q.checkpoint_after;
    document.getElementById("question-prompt").textContent = q.prompt;
    document.getElementById("answer-text").value = "";
    document.getElementById("validation-msg").style.display = "none";
    document.getElementById("next-btn").disabled = true;
    startTimer();
  }

  function hideQuestion() {
    answering = false;
    setCheckboxes(false);
    document.getElementById("question-panel").classList.remove("active");
    document.getElementById("next-btn").disabled = false;
    stopTimer();
  }

  // Exposed globally
  window.nextEpisode = async function() {
    var btn = document.getElementById("next-btn");
    btn.disabled = true;
    try {
      var data = await api("POST", "/api/next-episode");

      // Add the new episode to our list
      if (data.episode) {
        allEpisodes.push(data.episode);
        renderEpisodeFeed();
      }

      // Reload state for progress
      await loadState();

      if (data.checkpoint_triggered && data.pending_questions && data.pending_questions.length > 0) {
        currentQuestions = data.pending_questions;
        currentQuestionIndex = 0;
        showQuestion(currentQuestions[0]);
      } else {
        btn.disabled = false;
      }
    } catch(e) {
      alert("Error: " + e.message);
      btn.disabled = false;
    }
  };

  window.submitAnswer = async function() {
    var text = document.getElementById("answer-text").value.trim();
    var refs = getCheckedRefs();
    var msg = document.getElementById("validation-msg");

    if (!text || refs.length === 0) {
      msg.style.display = "block";
      return;
    }
    msg.style.display = "none";

    var wallMs = stopTimer();
    var q = currentQuestions[currentQuestionIndex];
    var btn = document.getElementById("submit-btn");
    btn.disabled = true;

    try {
      await api("POST", "/api/submit-answer", {
        question_id: q.question_id,
        answer_text: text,
        refs_cited: refs,
        wall_time_ms: wallMs
      });

      currentQuestionIndex++;
      if (currentQuestionIndex < currentQuestions.length) {
        showQuestion(currentQuestions[currentQuestionIndex]);
      } else {
        hideQuestion();
        await loadState();
      }
    } catch(e) {
      alert("Error: " + e.message);
    } finally {
      btn.disabled = false;
    }
  };

  window.finishBenchmark = async function() {
    var btn = document.getElementById("finish-btn");
    btn.disabled = true;
    try {
      var data = await api("POST", "/api/finish");
      var res = document.getElementById("finish-result");
      res.innerHTML = '<div class="output-path">' + esc(data.output_path) + '</div>'
        + '<div class="instructions">'
        + '<p>Next steps:</p>'
        + '<p><code>lens score --run ' + esc(data.output_path) + '</code></p>'
        + '<p><code>lens report --run ' + esc(data.output_path) + ' --format html</code></p>'
        + '</div>';
    } catch(e) {
      alert("Error: " + e.message);
      btn.disabled = false;
    }
  };

  // Init
  (async function() {
    await loadState();
    await loadEpisodes();
  })();

})();
</script>
</body>
</html>"""
