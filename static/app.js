/* Profile Photo Generator - gallery UI.
 *
 * Deliberately plain: no framework, no build step, no CDN. The form is built
 * from GET /v1/options at load time rather than hard-coded, so editing
 * vocab.yaml immediately changes the UI and the two can never drift apart.
 */

const $ = (id) => document.getElementById(id);

const state = {
  options: null,
  // Axes shown in the "Who" block; everything else goes under "Look".
  primary: ['sex', 'ethnicity', 'skin_tone', 'profession'],
  secondary: ['hair', 'facial_hair', 'glasses', 'expression', 'clothing', 'background', 'lighting'],
  lastAttributes: null,
  count: 0,
};

/* ---------------------------------------------------------------- helpers */

async function api(path, init) {
  const response = await fetch(path, init);
  if (response.status === 204) return null;
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
    } catch { /* response had no JSON body */ }
    throw new Error(detail);
  }
  return response.json();
}

function say(message, isError = false) {
  const hint = $('hint');
  hint.textContent = message || '';
  hint.classList.toggle('error', Boolean(isError));
}

function titleCase(text) {
  return text.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase());
}

/* ------------------------------------------------------------------- form */

function buildSelect(axis, options) {
  const label = document.createElement('label');
  const span = document.createElement('span');
  span.textContent = titleCase(axis);

  const select = document.createElement('select');
  select.id = `axis-${axis}`;
  select.dataset.axis = axis;

  const any = document.createElement('option');
  any.value = '';
  any.textContent = 'any';
  select.append(any);

  for (const option of options) {
    const element = document.createElement('option');
    element.value = option.value;
    element.textContent = option.label || option.value;
    select.append(element);
  }

  label.append(span, select);
  return label;
}

function buildForm() {
  const { axes } = state.options;
  const place = (ids, container) => {
    container.replaceChildren();
    for (const axis of ids) {
      if (axes[axis]) container.append(buildSelect(axis, axes[axis]));
    }
  };
  place(state.primary, $('axes-primary'));
  place(state.secondary, $('axes-secondary'));

  const bounds = state.options.age_bounds;
  const age = $('age');
  age.min = bounds.min;
  age.max = bounds.max;
}

function readForm() {
  const payload = {};
  for (const select of document.querySelectorAll('select[data-axis]')) {
    if (select.value) payload[select.dataset.axis] = select.value;
  }
  if ($('age-enabled').checked) payload.age = Number($('age').value);
  if ($('seed').value.trim()) payload.seed = $('seed').value.trim();
  if ($('prompt_extra').value.trim()) payload.prompt_extra = $('prompt_extra').value.trim();
  if ($('fast').checked) payload.fast = true;
  return payload;
}

function applyAttributes(attributes) {
  for (const select of document.querySelectorAll('select[data-axis]')) {
    const value = attributes[select.dataset.axis];
    select.value = value && [...select.options].some((o) => o.value === value) ? value : '';
  }
  if (attributes.age) {
    $('age-enabled').checked = true;
    $('age').disabled = false;
    $('age').value = attributes.age;
    $('age-out').textContent = attributes.age;
  }
}

/* ---------------------------------------------------------------- gallery */

function card(avatar) {
  // A button, not a div: the card opens a dialog, so it has to be reachable
  // and activatable from the keyboard. Using the real element gets focus,
  // Enter and Space for free rather than reimplementing them.
  const element = document.createElement('button');
  element.type = 'button';
  element.className = 'card';
  element.dataset.id = avatar.id;
  element.setAttribute(
    'aria-label',
    avatar.persona
      ? `Details for ${avatar.persona.name}, ${avatar.persona.age}`
      : 'Avatar details',
  );

  const img = document.createElement('img');
  img.src = `/v1/avatars/${avatar.id}/image?size=256`;
  img.alt = avatar.persona ? `Portrait of ${avatar.persona.name}` : 'Generated portrait';
  img.loading = 'lazy';

  const caption = document.createElement('div');
  caption.className = 'cap';
  caption.textContent = avatar.persona
    ? `${avatar.persona.name}, ${avatar.persona.age}`
    : `${avatar.attributes.age} · ${avatar.attributes.sex}`;

  // Nested inside a button would be invalid HTML, so the delete control is a
  // sibling and the pair share a wrapper.
  const del = document.createElement('button');
  del.className = 'del';
  del.type = 'button';
  del.textContent = '×';
  del.title = 'Delete this avatar';
  del.setAttribute('aria-label', 'Delete this avatar');
  del.addEventListener('click', () => deleteAvatar(avatar.id, wrapper));

  element.append(img, caption);
  element.addEventListener('click', () => openDetail(avatar));

  const wrapper = document.createElement('div');
  wrapper.className = 'card-wrap';
  wrapper.dataset.id = avatar.id;
  wrapper.append(element, del);
  return wrapper;
}

async function deleteAvatar(id, element) {
  try {
    await api(`/v1/avatars/${id}`, { method: 'DELETE' });
    element?.remove();
    updateCount(-1);
    say('Deleted.');
  } catch (error) {
    say(`Could not delete: ${error.message}`, true);
  }
}

/* Two-step confirmation in the button itself rather than window.confirm():
   a native dialog blocks the page, and this keeps the warning next to the
   thing it is warning about. Disarms itself after five seconds. */
function armable(button, label, armedLabel, action) {
  let armed = false;
  let timer = null;
  const disarm = () => {
    armed = false;
    button.textContent = label;
    button.classList.remove('armed');
    clearTimeout(timer);
  };
  button.addEventListener('click', async () => {
    if (!armed) {
      armed = true;
      button.textContent = armedLabel();
      button.classList.add('armed');
      timer = setTimeout(disarm, 5000);
      return;
    }
    disarm();
    await action();
  });
  return disarm;
}

function pendingCard(label = 'generating…') {
  const element = document.createElement('div');
  element.className = 'card pending';
  element.textContent = label;
  return element;
}

async function refreshGallery() {
  const avatars = await api('/v1/avatars?limit=60');
  const gallery = $('gallery');
  gallery.replaceChildren(...avatars.map(card));
  $('empty').hidden = avatars.length > 0;
  state.count = avatars.length;
  renderCount();
  return avatars;
}

function renderCount() {
  $('count').textContent = state.count ? `· ${state.count}` : '';
  $('clear').disabled = !state.count;
}

function updateCount(delta) {
  state.count = Math.max(0, (state.count || 0) + delta);
  renderCount();
  $('empty').hidden = state.count > 0;
}

/* ----------------------------------------------------------------- detail */

function openDetail(avatar) {
  state.lastAttributes = avatar.attributes;

  $('detail-img').src = `/v1/avatars/${avatar.id}/image?size=512`;
  $('detail-name').textContent = avatar.persona ? avatar.persona.name : 'Generated portrait';
  $('detail-sub').textContent = avatar.persona
    ? [avatar.persona.occupation, avatar.persona.city].filter(Boolean).join(' · ')
    : '';

  const list = $('detail-attrs');
  list.replaceChildren();
  const shown = { ...avatar.attributes, seed: avatar.seed, composer: avatar.composer };
  if (avatar.seed_key) shown.seed_key = avatar.seed_key;
  for (const [key, value] of Object.entries(shown)) {
    const dt = document.createElement('dt');
    dt.textContent = titleCase(key);
    const dd = document.createElement('dd');
    dd.textContent = String(value).replace(/_/g, ' ');
    list.append(dt, dd);
  }

  $('detail-prompt').textContent = avatar.prompt;
  $('copy-url').onclick = () => {
    const url = `${location.origin}/v1/avatars/${avatar.id}/image?size=256`;
    navigator.clipboard.writeText(url).then(
      () => ($('copy-url').textContent = 'Copied'),
      () => ($('copy-url').textContent = url),
    );
    setTimeout(() => ($('copy-url').textContent = 'Copy image URL'), 1600);
  };
  $('reroll').onclick = () => {
    $('detail').close();
    applyAttributes(avatar.attributes);
    $('seed').value = '';
    say('Traits loaded. Generate for a new face with the same traits.');
  };

  // Replaced wholesale each time the dialog opens, so the handler always
  // refers to the avatar currently on screen.
  const del = $('delete-one');
  const fresh = del.cloneNode(true);
  del.replaceWith(fresh);
  armable(fresh, 'Delete', () => 'Really delete?', async () => {
    $('detail').close();
    await deleteAvatar(avatar.id, document.querySelector(`.card-wrap[data-id="${avatar.id}"]`));
  });

  $('detail').showModal();
}

/* -------------------------------------------------------------- generation */

async function generate() {
  const button = $('go');
  button.disabled = true;
  say('Generating…');

  const placeholder = pendingCard();
  $('gallery').prepend(placeholder);
  $('empty').hidden = true;

  try {
    const result = await api('/v1/avatars', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(readForm()),
    });
    placeholder.replaceWith(card(result));
    updateCount(1);
    const seconds = result.duration_ms ? ` in ${(result.duration_ms / 1000).toFixed(1)}s` : '';
    say(result.cached ? 'Already generated - served from cache.' : `Done${seconds}.`);
  } catch (error) {
    placeholder.remove();
    say(error.message, true);
  } finally {
    button.disabled = false;
  }
}

async function generateBatch() {
  const count = Math.max(1, Math.min(100, Number($('batch-n').value) || 12));
  const button = $('batch');
  button.disabled = true;
  say(`Queued ${count}…`);

  const placeholders = Array.from({ length: count }, () => pendingCard('queued'));
  $('gallery').prepend(...placeholders);
  $('empty').hidden = true;

  try {
    const job = await api('/v1/avatars/batch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ n: count, diversity: 'even', overrides: readForm() }),
    });
    await pollJob(job.id, placeholders);
  } catch (error) {
    placeholders.forEach((p) => p.remove());
    say(error.message, true);
  } finally {
    button.disabled = false;
  }
}

async function pollJob(jobId, placeholders) {
  // Poll rather than stream: generation is measured in seconds, so a two
  // second interval is plenty and keeps the server free of open connections.
  for (;;) {
    const job = await api(`/v1/jobs/${jobId}`);
    const eta = job.eta_seconds ? `, about ${Math.round(job.eta_seconds)}s left` : '';
    say(`${job.completed}/${job.total} done${eta}`);

    if (job.status === 'done' || job.status === 'failed') {
      placeholders.forEach((p) => p.remove());
      await refreshGallery();
      say(job.error ? `Finished with errors: ${job.error}` : `Generated ${job.completed}.`, Boolean(job.error));
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
}

/* ------------------------------------------------------------------ status */

async function pollStatus() {
  const dot = $('status-dot');
  try {
    const health = await api('/readyz');
    dot.className = `dot ${health.status === 'ready' ? 'ready' : health.status === 'error' ? 'error' : 'loading'}`;
    dot.title = health.status;
    const bits = [health.backend, health.device];
    if (health.queue_depth) bits.push(`queue ${health.queue_depth}`);
    if (health.ollama_reachable === false) bits.push('ollama off (template prompts)');
    $('server-meta').textContent = bits.join(' · ');
  } catch {
    dot.className = 'dot error';
    $('server-meta').textContent = 'server unreachable';
  }
}

/* -------------------------------------------------------------------- init */

async function init() {
  $('age-enabled').addEventListener('change', (event) => {
    $('age').disabled = !event.target.checked;
    $('age-out').textContent = event.target.checked ? $('age').value : 'any';
  });
  $('age').addEventListener('input', (event) => ($('age-out').textContent = event.target.value));

  $('generator').addEventListener('submit', (event) => {
    event.preventDefault();
    generate();
  });
  $('randomise').addEventListener('click', () => {
    document.querySelectorAll('select[data-axis]').forEach((s) => (s.value = ''));
    $('age-enabled').checked = false;
    $('age').disabled = true;
    $('age-out').textContent = 'any';
    $('seed').value = '';
    say('Cleared. Everything will be randomised.');
  });
  $('batch').addEventListener('click', generateBatch);
  $('refresh').addEventListener('click', () => refreshGallery().then(() => say('')));

  armable(
    $('clear'),
    'Clear all',
    () => `Delete all ${state.count}?`,
    async () => {
      try {
        const result = await api('/v1/avatars?confirm=true', { method: 'DELETE' });
        await refreshGallery();
        say(`Deleted ${result.deleted}.`);
      } catch (error) {
        say(`Could not clear: ${error.message}`, true);
      }
    },
  );
  $('detail-close').addEventListener('click', () => $('detail').close());

  try {
    state.options = await api('/v1/options');
    buildForm();
  } catch (error) {
    say(`Could not load options: ${error.message}`, true);
  }

  await refreshGallery().catch(() => {});
  pollStatus();
  setInterval(pollStatus, 5000);
}

init();
