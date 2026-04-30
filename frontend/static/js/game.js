/* ═══════════════════════════════════════════════════════════════════════
   EverWrite — Frontend Game Logic
   ═══════════════════════════════════════════════════════════════════════ */

'use strict';

// ── State ────────────────────────────────────────────────────────────────
let sessionId   = null;
let isStreaming = false;

// Spinner frames
const SPINNER = ['|', '/', '—', '\\'];
let   spinIdx  = 0;
let   spinTimer = null;
const API_BASE_URL = (window.EVERWRITE_CONFIG?.API_BASE_URL || '').replace(/\/$/, '');

// ── DOM references ───────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const narrativeEl  = $('narrative-area');
const playerInput  = $('player-input');
const btnSend      = $('btn-send');
const hudPhase     = $('hud-phase');
const hudCharacter = $('hud-character');
const hudFaction   = $('hud-faction');
const hudEquip     = $('hud-equip');
const hudHealth    = $('hud-health');
const hudAttune    = $('hud-attunement');
const hudInfluence = $('hud-influence');
const hudKnowledge = $('hud-knowledge');
const gameUi       = $('game-ui');
const startScreen  = $('start-screen');

// ═══════════════════════════════════════════════════════════════════════
// PUBLIC API (called from HTML)
// ═══════════════════════════════════════════════════════════════════════

/** Called by the "Begin Journey" button on the start screen. */
async function startGame() {
  const btnStart = $('btn-start');
  btnStart.disabled = true;

  // Transition: fade start screen → reveal game UI
  startScreen.classList.add('fade-out');
  setTimeout(() => {
    startScreen.style.display = 'none';
    gameUi.classList.remove('hidden');
    playerInput.focus();
  }, 550);

  sysMsg('◈  INITIALISING RESONANCE ENGINE  ◈');

  const loaderId = showLoader();
  try {
    const response = await fetch(`${API_BASE_URL}/api/start`, { method: 'POST' });
    hideLoader(loaderId);

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      sysMsg(`ERROR: ${err.error || 'Could not connect to the Aetheric Field.'}`);
      return;
    }

    await consumeStream(response, 'ai');
  } catch (err) {
    hideLoader(loaderId);
    sysMsg('ERROR: NETWORK FAILURE. CHECK YOUR CONNECTION AND API KEY.');
    console.error(err);
  }
}

/** Called by the Send button and Enter key. */
async function sendMessage() {
  if (isStreaming || !sessionId) return;

  const text = playerInput.value.trim();
  if (!text) return;

  playerInput.value = '';
  appendMsg(text, 'user');
  lockInput(true);

  const loaderId = showLoader();
  try {
    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ session_id: sessionId, message: text }),
    });

    hideLoader(loaderId);

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      sysMsg(`ERROR: ${err.error || 'Unknown error.'}`);
      lockInput(false);
      return;
    }

    await consumeStream(response, 'ai');
  } catch (err) {
    hideLoader(loaderId);
    sysMsg('ERROR: CONNECTION LOST TO THE AETHERIC FIELD.');
    console.error(err);
    lockInput(false);
  }
}

/** Restart button — reloads the page for a fresh session. */
function newGame() {
  if (!confirm('START A NEW JOURNEY? CURRENT PROGRESS WILL BE LOST.')) return;
  location.reload();
}

// ═══════════════════════════════════════════════════════════════════════
// SSE STREAM CONSUMER
// ═══════════════════════════════════════════════════════════════════════

/**
 * Reads an SSE-formatted streaming response and renders each chunk
 * character by character into a new message bubble.
 */
async function consumeStream(response, role) {
  isStreaming = true;

  // Create an empty message bubble with a blinking cursor
  const bubble = appendMsg('', role);
  const cursor = document.createElement('span');
  cursor.className = 'stream-cursor';
  bubble.appendChild(cursor);

  const reader  = response.body.getReader();
  const decoder = new TextDecoder();
  let   lineBuf = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      lineBuf += decoder.decode(value, { stream: true });
      const lines = lineBuf.split('\n');
      lineBuf = lines.pop();           // keep potentially incomplete last line

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (!raw) continue;

        let payload;
        try { payload = JSON.parse(raw); } catch { continue; }

        if (!payload.done && payload.chunk) {
          // Inject chunk text just before the cursor
          cursor.insertAdjacentText('beforebegin', payload.chunk);
          scrollBottom();
        }

        if (payload.done) {
          cursor.remove();

          if (payload.error) {
            // Show API / server error inside the bubble
            bubble.insertAdjacentText('beforeend', `\n\n⚠ ${payload.error}`);
            bubble.style.color = 'var(--red)';
          }

          if (payload.session_id) {
            sessionId = payload.session_id;
          }

          if (payload.state) {
            updateHUD(payload.state);
          }

          if (payload.state?.phase === 'ended') {
            sysMsg('══════  THE JOURNEY ENDS  ══  REFRESH TO WEAVE ANEW  ══════');
            lockInput(true);
          }
        }
      }
    }

    // Process any remaining buffer content
    if (lineBuf.startsWith('data: ')) {
      try {
        const payload = JSON.parse(lineBuf.slice(6).trim());
        if (payload.done && payload.session_id) sessionId = payload.session_id;
        if (payload.done && payload.state)      updateHUD(payload.state);
      } catch { /* ignore */ }
    }

  } finally {
    cursor.remove();             // safety: remove cursor even on error
    isStreaming = false;
    if (response.ok) lockInput(false);
  }
}

// ═══════════════════════════════════════════════════════════════════════
// HUD
// ═══════════════════════════════════════════════════════════════════════

function updateHUD(state) {
  const phase = (state.phase || 'intro');

  hudPhase.textContent = phase.toUpperCase();
  hudPhase.className   = `hud-value phase-${phase}`;

  hudCharacter.textContent = state.character_name ? clip(state.character_name, 15) : '—';
  hudFaction.textContent   = state.faction   ? clip(state.faction, 20)   : 'UNKNOWN';
  hudEquip.textContent     = state.equipment ? clip(state.equipment, 20) : 'NONE';
  
  // Additional attributes
  if (hudHealth)    hudHealth.textContent    = (state.health !== undefined) ? `${state.health}/10` : '—';
  if (hudAttune)    hudAttune.textContent    = (state.aetheric_attunement !== undefined) ? `${state.aetheric_attunement}/10` : '—';
  if (hudInfluence) hudInfluence.textContent = (state.influence !== undefined) ? `${state.influence}/10` : '—';
  if (hudKnowledge) hudKnowledge.textContent = (state.knowledge !== undefined) ? `${state.knowledge}/10` : '—';

  // Optionally show inventory/relations as system messages in HUD area
  if (state.inventory && Array.isArray(state.inventory)) {
    // keep a short summary in equipment slot if equipment is empty
    if (!state.equipment) hudEquip.textContent = state.inventory.slice(0,3).join(', ') || 'NONE';
  }
}

// ═══════════════════════════════════════════════════════════════════════
// DOM HELPERS
// ═══════════════════════════════════════════════════════════════════════

/** Append a narrative bubble and return the element. */
function appendMsg(text, role) {
  const div = document.createElement('div');
  div.className   = `message ${role}`;
  div.textContent = text;
  narrativeEl.appendChild(div);
  scrollBottom();
  return div;
}

/** Append a centred system / status notification. */
function sysMsg(text) {
  const div = document.createElement('div');
  div.className   = 'message system';
  div.textContent = text;
  narrativeEl.appendChild(div);
  scrollBottom();
}

/** Show a spinner row while waiting for the stream to begin. */
function showLoader() {
  const id  = `ldr-${Date.now()}`;
  const row = document.createElement('div');
  row.id        = id;
  row.className = 'loading-msg';

  const sp = document.createElement('span');
  sp.className       = 'spinner';
  sp.dataset.frame   = SPINNER[0];
  row.appendChild(sp);
  row.appendChild(document.createTextNode('\u00a0WEAVING RESONANCE RESPONSE...'));
  narrativeEl.appendChild(row);
  scrollBottom();

  spinTimer = setInterval(() => {
    spinIdx = (spinIdx + 1) % SPINNER.length;
    if (sp.isConnected) sp.dataset.frame = SPINNER[spinIdx];
  }, 110);

  return id;
}

function hideLoader(id) {
  clearInterval(spinTimer);
  const el = document.getElementById(id);
  if (el) el.remove();
}

/** Lock or unlock the input area. */
function lockInput(locked) {
  playerInput.disabled = locked;
  btnSend.disabled     = locked;
  if (!locked) playerInput.focus();
}

function scrollBottom() {
  narrativeEl.scrollTop = narrativeEl.scrollHeight;
}

function clip(str, max) {
  return str.length > max ? str.slice(0, max) + '…' : str;
}

// ═══════════════════════════════════════════════════════════════════════
// KEYBOARD SHORTCUTS
// ═══════════════════════════════════════════════════════════════════════

document.addEventListener('keydown', e => {
  // Enter in the input field → send
  if (e.key === 'Enter' && !e.shiftKey && document.activeElement === playerInput) {
    e.preventDefault();
    sendMessage();
  }
});
