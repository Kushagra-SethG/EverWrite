import { useEffect, useMemo, useRef, useState } from 'react';

const SPINNER = ['|', '/', '—', '\\'];
const API_BASE_URL = (window.EVERWRITE_CONFIG?.API_BASE_URL || '').replace(/\/$/, '');

const DEFAULT_STATE = {
  phase: 'name',
  character_name: '',
  faction: '',
  equipment: '',
  health: 10,
  attunement: 0,
  influence: 0,
  knowledge: 0,
  inventory: [],
  relationships: {},
  locations: [],
  npcs: [],
  events_count: 0,
};

// Faction data structure for frontend display
const FACTION_DATA = {
  "Lumina Concordat": {
    perks: ["Light Barrier (25% damage reduction)", "Healing Resonance (self-heal 1/turn)", "Ally of Healers (+2 influence)"],
    restraints: ["Pacifist reputation (-1 combat favor)", "Slower attunement growth"],
    philosophy: "Unity, Protection, Restoration",
    bonus: "+2 Knowledge"
  },
  "Cogwheel Dominion": {
    perks: ["Mechanical Armor (+1 health)", "Tech Gadget (puzzle utility)", "Engineer's Intuition (+1 trap detection)"],
    restraints: ["Magic weakness (-1 attunement cap)", "Distrusted by nature factions"],
    philosophy: "Progress, Control, Innovation",
    bonus: "+2 Influence"
  },
  "Verdant Enclave": {
    perks: ["Nature's Blessing (improved herb harvesting)", "Beast Companion (temporary ally)", "Growth Resonance (+1 per quest)"],
    restraints: ["Urban weakness (-1 influence in cities)", "Slow tech adaptation"],
    philosophy: "Balance, Nature, Harmony",
    bonus: "+2 Attunement"
  },
  "Obsidian Vault": {
    perks: ["Shadow Veil (hide from enemies)", "Dark Pact (risky stat boost)", "Greed (find rare items)"],
    restraints: ["Distrusted by most (-2 relations)", "Dark pacts cause unpredictable consequences"],
    philosophy: "Power, Ambition, Knowledge",
    bonus: "+2 Knowledge"
  },
  "Ashfall Clans": {
    perks: ["Battle Hardened (health regen in combat)", "Warlord Presence (intimidation)", "Volcanic Resonance (fire abilities)"],
    restraints: ["Diplomatic weakness (-1 with peaceful factions)", "Poor stealth and subterfuge"],
    philosophy: "Strength, Honor, Victory",
    bonus: "+2 Health"
  }
};

function normalizeState(next, prev = DEFAULT_STATE) {
  const incoming = next && typeof next === 'object' ? next : {};
  const merged = { ...DEFAULT_STATE, ...prev, ...incoming };
  return {
    ...merged,
    inventory: Array.isArray(merged.inventory) ? merged.inventory : [],
    relationships: merged.relationships && typeof merged.relationships === 'object' ? merged.relationships : {},
    locations: Array.isArray(merged.locations) ? merged.locations : [],
    npcs: Array.isArray(merged.npcs) ? merged.npcs : [],
    events_count: Number.isFinite(Number(merged.events_count)) ? Number(merged.events_count) : 0,
    health: Number.isFinite(Number(merged.health)) ? Number(merged.health) : 0,
    attunement: Number.isFinite(Number(merged.attunement)) ? Number(merged.attunement) : 0,
    influence: Number.isFinite(Number(merged.influence)) ? Number(merged.influence) : 0,
    knowledge: Number.isFinite(Number(merged.knowledge)) ? Number(merged.knowledge) : 0,
  };
}

function clampText(value, fallback = '—') {
  return value ? value : fallback;
}

function formatStat(value) {
  if (value === undefined || value === null) return '—';
  return `${value}/10`;
}

function statBar(value, max = 10) {
  const safe = Math.max(0, Math.min(max, Number(value) || 0));
  return `${'█'.repeat(safe)}${'░'.repeat(max - safe)}`;
}

function useInterval(callback, delay) {
  const savedCallback = useRef(callback);
  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);
  useEffect(() => {
    if (delay === null) return undefined;
    const id = setInterval(() => savedCallback.current(), delay);
    return () => clearInterval(id);
  }, [delay]);
}

// Component to show stat change indicator
function StatChange({ oldValue, newValue, label }) {
  if (oldValue === newValue) return null;
  const delta = newValue - oldValue;
  const sign = delta > 0 ? '+' : '';
  const className = delta > 0 ? 'stat-change stat-change--positive' : 'stat-change stat-change--negative';
  return <span className={className}>{sign}{delta}</span>;
}

// Component to display faction perks and restraints
function FactionPanel({ faction }) {
  if (!faction || !FACTION_DATA[faction]) {
    return <div className="panel-copy muted">Select a clan to view perks and restraints</div>;
  }
  const data = FACTION_DATA[faction];
  return (
    <div className="faction-detail">
      <div className="faction-philosophy">
        <strong>Creed:</strong> {data.philosophy}
      </div>
      <div className="faction-bonus">
        <strong>Starting Bonus:</strong> {data.bonus}
      </div>
      <div className="faction-perks">
        <h3>Perks & Abilities</h3>
        {data.perks.map((perk, i) => (
          <div key={i} className="perk-item">✦ {perk}</div>
        ))}
      </div>
      <div className="faction-restraints">
        <h3>Restraints & Weaknesses</h3>
        {data.restraints.map((restraint, i) => (
          <div key={i} className="restraint-item">✕ {restraint}</div>
        ))}
      </div>
    </div>
  );
}

function App() {
  const [sessionId, setSessionId] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [state, setState] = useState(DEFAULT_STATE);
  const [prevState, setPrevState] = useState(DEFAULT_STATE);
  const [loadingLabel, setLoadingLabel] = useState('AWAKENING IN THE THRONE OF REBIRTH...');
  const [spinIdx, setSpinIdx] = useState(0);
  const [activeTab, setActiveTab] = useState('character');
  const [statChangeHighlight, setStatChangeHighlight] = useState(null);
  const inputRef = useRef(null);
  const narrativeRef = useRef(null);

  useInterval(() => setSpinIdx((idx) => (idx + 1) % SPINNER.length), 120);

  const apiBase = useMemo(() => API_BASE_URL, []);

  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = '0px';
    el.style.height = `${el.scrollHeight}px`;
  }, [input]);

  function scrollBottom() {
    const el = narrativeRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }

  function pushMessage(role, text, status = 'normal') {
    setMessages((prev) => [...prev, { role, text, status, id: `${Date.now()}-${Math.random()}` }]);
  }

  async function consumeStream(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let currentAiMessageId = null;
    setIsStreaming(true);

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6).trim();
          if (!raw) continue;

          let payload;
          try {
            payload = JSON.parse(raw);
          } catch {
            continue;
          }

          if (!payload.done && payload.chunk) {
            setMessages((prev) => {
              const next = [...prev];
              if (currentAiMessageId === null) {
                currentAiMessageId = `${Date.now()}-ai-${Math.random()}`;
                next.push({ role: 'ai', text: payload.chunk, status: 'streaming', id: currentAiMessageId });
              } else {
                const idx = next.findIndex((msg) => msg && msg.id === currentAiMessageId);
                if (idx >= 0 && next[idx]) {
                  next[idx] = {
                    ...next[idx],
                    text: `${next[idx].text || ''}${payload.chunk}`,
                  };
                } else {
                  // Safety fallback in case a previous state update removed/replaced the streaming message.
                  currentAiMessageId = `${Date.now()}-ai-${Math.random()}`;
                  next.push({ role: 'ai', text: payload.chunk, status: 'streaming', id: currentAiMessageId });
                }
              }
              return next;
            });
            continue;
          }

          if (payload.session_id) {
            setSessionId(payload.session_id);
          }
          if (payload.state) {
            setState((prev) => {
              const next = normalizeState(payload.state, prev);
              // Track stat changes for visual feedback
              if (next.health !== prev.health || next.attunement !== prev.attunement || 
                  next.influence !== prev.influence || next.knowledge !== prev.knowledge) {
                setStatChangeHighlight(Date.now());
                setTimeout(() => setStatChangeHighlight(null), 1500);
              }
              setPrevState(prev);
              return next;
            });
          }
          if (payload.error) {
            pushMessage('system', `⚠ ${payload.error}`, 'error');
          }
          if (payload.done) {
            setMessages((prev) => prev.map((msg) => (msg.status === 'streaming' ? { ...msg, status: 'normal' } : msg)));
          }
        }
      }
    } finally {
      setIsStreaming(false);
      setMessages((prev) => prev.map((msg) => (msg.status === 'streaming' ? { ...msg, status: 'normal' } : msg)));
      inputRef.current?.focus();
    }
  }

  async function startGame() {
    pushMessage('system', '◈ INITIALISING RESONANCE ENGINE ◈');
    setLoadingLabel('WEAVING YOUR REBIRTH...');
    setIsStreaming(true);

    try {
      const response = await fetch(`${apiBase}/api/start`, { method: 'POST' });
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        pushMessage('system', `ERROR: ${err.error || 'Could not awaken the field.'}`, 'error');
        return;
      }
      await consumeStream(response);
    } catch (error) {
      pushMessage('system', 'ERROR: CONNECTION LOST TO THE AETHERIC FIELD.', 'error');
      console.error(error);
    }
  }

  async function sendMessage(event) {
    event?.preventDefault();
    const text = input.trim();
    if (!text || isStreaming || !sessionId) return;

    setInput('');
    pushMessage('user', text);
    setLoadingLabel('FATE IS RESPONDING...');

    try {
      const response = await fetch(`${apiBase}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message: text }),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        pushMessage('system', `ERROR: ${err.error || 'Unknown error.'}`, 'error');
        return;
      }

      await consumeStream(response);
    } catch (error) {
      pushMessage('system', 'ERROR: CONNECTION LOST TO THE AETHERIC FIELD.', 'error');
      console.error(error);
    }
  }

  function resetGame() {
    if (!window.confirm('START A NEW JOURNEY? CURRENT PROGRESS WILL BE LOST.')) return;
    window.location.reload();
  }

  function handleInputKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  function handleInputChange(event) {
    const el = event.target;
    setInput(el.value);
    el.style.height = '0px';
    el.style.height = `${el.scrollHeight}px`;
  }

  const recentRelations = useMemo(() => Object.entries(state.relationships || {}), [state.relationships]);

  return (
    <div className="app-shell">
      <div className="ambient-bg" />
      <div className="ambient-bg ambient-bg--alt" />
      <div className="vine-frame vine-frame--top" />
      <div className="vine-frame vine-frame--bottom" />

      <header className="topbar">
        <div>
          <div className="brand-mark">EVERWRITE</div>
          <div className="brand-sub">A reincarnation fantasy in Aethel</div>
        </div>
        <div className="topbar-stats">
          <div className="stat-pill"><span>PHASE</span><strong>{clampText(state.phase).toUpperCase()}</strong></div>
          <div className="stat-pill"><span>NAME</span><strong>{clampText(state.character_name)}</strong></div>
          <div className="stat-pill"><span>CLAN</span><strong>{clampText(state.faction)}</strong></div>
          <button className="ghost-btn" onClick={resetGame}>New Life</button>
        </div>
      </header>

      <section className="hero-panel">
        <div className="hero-copy">
          <p className="eyebrow">Reborn Soul // Clan Chronicle</p>
          <h1>Write your fate in a world of bloodlines, magic, and old vows.</h1>
          <p className="hero-text">
            Choose a name, be born into a clan, and let every decision reshape your stats, allies, and destiny.
          </p>
          <div className="hero-actions">
            <button className="primary-btn" onClick={startGame} disabled={isStreaming && messages.length > 0}>
              Begin Rebirth
            </button>
            <div className="loader-line">
              <span className="spinner">{SPINNER[spinIdx]}</span>
              <span>{loadingLabel}</span>
            </div>
          </div>
        </div>

        <aside className="stat-cards">
          <div className="card card--glow">
            <div className="card-title">Health</div>
            <div className="card-value">{formatStat(state.health)}</div>
            <div className="card-bar">{statBar(state.health)}</div>
          </div>
          <div className="card">
            <div className="card-title">Attunement</div>
            <div className="card-value">{formatStat(state.attunement)}</div>
            <div className="card-bar">{statBar(state.attunement)}</div>
          </div>
          <div className="card">
            <div className="card-title">Influence</div>
            <div className="card-value">{formatStat(state.influence)}</div>
            <div className="card-bar">{statBar(state.influence)}</div>
          </div>
          <div className="card">
            <div className="card-title">Knowledge</div>
            <div className="card-value">{formatStat(state.knowledge)}</div>
            <div className="card-bar">{statBar(state.knowledge)}</div>
          </div>
        </aside>
      </section>

      <main className="game-grid">
        <section className="narrative-panel">
          <button
            type="button"
            className="narrative-jump-btn"
            onClick={scrollBottom}
            aria-label="Jump to latest chat result"
            title="Jump to bottom"
          >
            ↓
          </button>
          <div className="narrative-scroll" ref={narrativeRef}>
            {messages.length === 0 ? (
              <div className="message system intro-message">
                Press <strong>Begin Rebirth</strong> to awaken, then choose your name, clan, and path.
              </div>
            ) : null}
            {messages.map((msg) => (
              <div key={msg.id} className={`message message--${msg.role} message--${msg.status}`}>
                {msg.text}
              </div>
            ))}
          </div>
        </section>

        <aside className="side-panel">
          {/* Tab Navigation */}
          <div className="tab-nav">
            <button
              className={`tab-btn ${activeTab === 'character' ? 'tab-btn--active' : ''}`}
              onClick={() => setActiveTab('character')}
            >
              CHARACTER
            </button>
            <button
              className={`tab-btn ${activeTab === 'perks' ? 'tab-btn--active' : ''}`}
              onClick={() => setActiveTab('perks')}
            >
              PERKS
            </button>
            <button
              className={`tab-btn ${activeTab === 'restraints' ? 'tab-btn--active' : ''}`}
              onClick={() => setActiveTab('restraints')}
            >
              RESTRAINTS
            </button>
          </div>

          {/* CHARACTER SHEET TAB */}
          {activeTab === 'character' && (
            <div className="panel-content">
              <div className="panel-section">
                <h2>Character Status</h2>
                <div className="character-info">
                  <div className="info-row">
                    <span>Name</span>
                    <strong>{clampText(state.character_name, 'Unnamed')}</strong>
                  </div>
                  <div className="info-row">
                    <span>Clan</span>
                    <strong>{clampText(state.faction, 'Unborn')}</strong>
                  </div>
                  <div className="info-row">
                    <span>Equipment</span>
                    <strong>{clampText(state.equipment, 'None')}</strong>
                  </div>
                </div>
              </div>

              <div className="panel-section">
                <h2>Stats</h2>
                <div className={`stat-entry ${statChangeHighlight && state.health !== prevState.health ? 'stat-entry--highlight' : ''}`}>
                  <div className="stat-label">
                    <span>Health</span>
                    <StatChange oldValue={prevState.health} newValue={state.health} />
                  </div>
                  <div className="stat-value">{formatStat(state.health)}</div>
                  <div className="stat-bar">{statBar(state.health)}</div>
                </div>

                <div className={`stat-entry ${statChangeHighlight && state.attunement !== prevState.attunement ? 'stat-entry--highlight' : ''}`}>
                  <div className="stat-label">
                    <span>Attunement</span>
                    <StatChange oldValue={prevState.attunement} newValue={state.attunement} />
                  </div>
                  <div className="stat-value">{formatStat(state.attunement)}</div>
                  <div className="stat-bar">{statBar(state.attunement)}</div>
                </div>

                <div className={`stat-entry ${statChangeHighlight && state.influence !== prevState.influence ? 'stat-entry--highlight' : ''}`}>
                  <div className="stat-label">
                    <span>Influence</span>
                    <StatChange oldValue={prevState.influence} newValue={state.influence} />
                  </div>
                  <div className="stat-value">{formatStat(state.influence)}</div>
                  <div className="stat-bar">{statBar(state.influence)}</div>
                </div>

                <div className={`stat-entry ${statChangeHighlight && state.knowledge !== prevState.knowledge ? 'stat-entry--highlight' : ''}`}>
                  <div className="stat-label">
                    <span>Knowledge</span>
                    <StatChange oldValue={prevState.knowledge} newValue={state.knowledge} />
                  </div>
                  <div className="stat-value">{formatStat(state.knowledge)}</div>
                  <div className="stat-bar">{statBar(state.knowledge)}</div>
                </div>
              </div>

              <div className="panel-section">
                <h2>Progress</h2>
                <div className="ledger-item"><span>Events</span><strong>{state.events_count || 0}</strong></div>
                <div className="ledger-item"><span>Locations</span><strong>{(state.locations || []).length}</strong></div>
                <div className="ledger-item"><span>NPCs Met</span><strong>{(state.npcs || []).length}</strong></div>
              </div>
            </div>
          )}

          {/* PERKS & ABILITIES TAB */}
          {activeTab === 'perks' && (
            <div className="panel-content">
              <div className="panel-section">
                <h2>Clan Perks & Abilities</h2>
                <FactionPanel faction={state.faction} />
              </div>
              {state.faction && (
                <div className="panel-section">
                  <h2>Relations</h2>
                  <div className="relations-list">
                    {Object.entries(state.relationships || {}).map(([name, value]) => (
                      <div className="relation-row" key={name}>
                        <span>{name}</span>
                        <strong className={value > 0 ? 'positive' : value < 0 ? 'negative' : ''}>
                          {value > 0 ? `+${value}` : value}
                        </strong>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* RESTRAINTS TAB */}
          {activeTab === 'restraints' && (
            <div className="panel-content">
              <div className="panel-section">
                <h2>Clan Restraints</h2>
                {state.faction && FACTION_DATA[state.faction] ? (
                  <div>
                    <h3>Weaknesses & Limitations</h3>
                    {FACTION_DATA[state.faction].restraints.map((restraint, i) => (
                      <div key={i} className="restraint-item full-restraint">
                        <span className="restraint-icon">⚠</span>
                        <span>{restraint}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="panel-copy muted">Select a clan to view restraints</div>
                )}
              </div>
              <div className="panel-section">
                <h2>Inventory</h2>
                <div className="inventory-list">
                  {(state.inventory || []).length ? state.inventory.map((item) => <span key={item} className="chip">{item}</span>) : <span className="muted">Nothing yet</span>}
                </div>
              </div>
            </div>
          )}
        </aside>
      </main>

      <div className="input-bar">
        <span className="input-caret">▶</span>
        <textarea
          ref={inputRef}
          value={input}
          onChange={handleInputChange}
          onKeyDown={handleInputKeyDown}
          className="player-input"
          placeholder={state.phase === 'name' ? 'Enter your new name...' : 'Type your choice, plan, or action...'}
          maxLength={500}
          rows={1}
          autoComplete="off"
          spellCheck="false"
          disabled={isStreaming && !sessionId}
        />
        <button className="send-btn" type="button" onClick={sendMessage} disabled={!input.trim() || isStreaming}>Send</button>
      </div>
    </div>
  );
}

export default App;
