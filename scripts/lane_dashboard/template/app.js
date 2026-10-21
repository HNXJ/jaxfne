(function () {
  const inRegistry = (r) => r.standing !== 'not in the registry';

  dash.calc('counts', {
    title: 'Lane counts',
    description: 'Counts over the lane rows.',
    inputs: ['lanes'],
    fn: (lanes) => [
      { metric: 'lanes', value: lanes.length },
      { metric: 'registry', value: lanes.filter(inRegistry).length },
      { metric: 'render', value: lanes.filter((r) => Boolean(r.rendered_utc)).length },
      { metric: 'receipts', value: lanes.filter((r) => Number(r.n_receipts) > 0).length },
      { metric: 'open', value: lanes.filter((r) => Number(r.n_open_items) > 0).length },
    ],
  });

  dash.calc('an_counts', {
    title: 'Saved-file counts',
    description: 'Counts and totals over the saved analysis files.',
    inputs: ['analyses'],
    fn: (analyses) => [
      { metric: 'files', value: analyses.length },
      { metric: 'areas', value: new Set(analyses.map((r) => r.area)).size },
      { metric: 'rows', value: analyses.reduce((s, r) => s + (r.n_rows === '' || r.n_rows == null ? 0 : Number(r.n_rows)), 0) },
      { metric: 'mb', value: Math.round(analyses.reduce((s, r) => s + Number(r.size_kb), 0) / 1024) },
      { metric: 'untracked', value: analyses.filter((r) => !r.tracked).length },
    ],
  });

  const state = { set: 'all', q: '', sort: 'name' };
  const grid = document.getElementById('fg-grid');
  const countEl = document.getElementById('fg-count');
  const fmtDate = d3.utcFormat('%Y-%m-%d');
  const parse = (s) => (s ? new Date(s) : null);
  const plural = (n, one, many) => n + ' ' + (Number(n) === 1 ? one : many);

  function pill(text, cls, title) {
    const s = document.createElement('span');
    s.className = 'pill' + (cls ? ' ' + cls : '');
    s.textContent = text;
    if (title) s.title = title;
    return s;
  }

  function card(r) {
    const c = document.createElement('article');
    c.className = 'card';
    c.id = 'card-' + r.id;
    if (r.thumb) {
      const img = document.createElement('img');
      img.alt = r.title + ' (thumbnail of ' + r.thumb_path + ')';
      img.loading = 'lazy';
      img.src = r.thumb;
      c.appendChild(img);
    } else {
      const n = document.createElement('div');
      n.className = 'noimg';
      n.textContent = 'No rendered figure in the lane';
      c.appendChild(n);
    }
    const b = document.createElement('div');
    b.className = 'body';
    const id = document.createElement('div');
    id.className = 'id';
    id.textContent = r.registry_ids ? r.id + ' · ' + r.registry_ids : r.id;
    const h = document.createElement('h3');
    h.textContent = r.title;
    const p = document.createElement('p');
    p.textContent = r.caption || 'No README, SOURCE_P1.md or NOTES.md in the lane.';
    if (r.caption_source) p.title = 'from ' + r.caption_source;
    const facts = document.createElement('div');
    facts.className = 'facts';
    facts.appendChild(pill(r.standing, inRegistry(r) ? 'ok' : 'warn'));
    if (r.release) facts.appendChild(pill('release ' + r.release));
    const d = parse(r.rendered_utc);
    if (d) facts.appendChild(pill('rendered ' + fmtDate(d) + (r.render_source === 'git' ? '' : ' (file time)'), r.render_source === 'git' ? '' : 'warn', r.thumb_path));
    const rd = parse(r.newest_receipt_utc);
    facts.appendChild(pill(plural(r.n_receipts, 'receipt', 'receipts') + (rd ? ', newest ' + fmtDate(rd) : ''), '', r.newest_receipt));
    facts.appendChild(pill(plural(r.n_data_files, 'data file', 'data files'), '', r.data_files));
    if (Number(r.n_open_items) > 0) facts.appendChild(pill(r.open_items, 'warn', 'open rows naming this lane'));
    b.append(id, h, p, facts);
    c.appendChild(b);
    return c;
  }

  function draw() {
    const res = dash.data('lanes');
    grid.replaceChildren();
    if (res.status !== 'ok') {
      grid.classList.toggle('dash-skeleton', res.status === 'loading');
      countEl.textContent = '';
      if (res.status === 'error') {
        const m = document.createElement('div');
        m.className = 'msg';
        m.textContent = 'lanes.json could not be read' + (res.error ? ': ' + res.error : '.');
        grid.appendChild(m);
      }
      return;
    }
    grid.classList.remove('dash-skeleton');
    const q = state.q.trim().toLowerCase();
    const keep = {
      all: () => true,
      registry: inRegistry,
      unregistered: (r) => !inRegistry(r),
      open: (r) => Number(r.n_open_items) > 0,
    };
    let rows = res.data.filter(keep[state.set])
      .filter((r) => !q || (r.id + ' ' + r.title + ' ' + r.caption + ' ' + r.registry_ids).toLowerCase().includes(q));
    const by = {
      name: (a, b) => a.id.localeCompare(b.id),
      render: (a, b) => String(b.rendered_utc).localeCompare(String(a.rendered_utc)) || by.name(a, b),
      receipt: (a, b) => String(b.newest_receipt_utc).localeCompare(String(a.newest_receipt_utc)) || by.name(a, b),
      open: (a, b) => Number(b.n_open_items) - Number(a.n_open_items) || by.name(a, b),
    };
    rows = rows.slice().sort(by[state.sort]);
    countEl.textContent = rows.length + ' of ' + res.data.length + ' lanes';
    if (!rows.length) {
      const m = document.createElement('div');
      m.className = 'msg';
      m.textContent = 'No lane matches. Clear the search or choose another set.';
      grid.appendChild(m);
      return;
    }
    const frag = document.createDocumentFragment();
    rows.forEach((r) => frag.appendChild(card(r)));
    grid.appendChild(frag);
  }

  document.querySelectorAll('.seg button[data-set]').forEach((btn) => {
    btn.addEventListener('click', () => {
      state.set = btn.dataset.set;
      document.querySelectorAll('.seg button[data-set]').forEach((x) => x.setAttribute('aria-pressed', String(x === btn)));
      draw();
    });
  });
  document.getElementById('fg-q').addEventListener('input', (e) => { state.q = e.target.value; draw(); });
  document.getElementById('fg-sort').addEventListener('change', (e) => { state.sort = e.target.value; draw(); });

  function td(text, cls) {
    const c = document.createElement('td');
    if (cls) c.className = cls;
    c.textContent = text;
    return c;
  }

  const SHOWN = 500;
  const an = { kind: 'all', q: '', sort: 'written' };
  const anBody = document.getElementById('an-body');
  const anCount = document.getElementById('an-count');
  const fmtInt = d3.format(',d');
  const fmtDay = d3.utcFormat('%Y-%m-%d %H:%M');

  function drawAnalyses() {
    const res = dash.data('analyses');
    anBody.replaceChildren();
    if (res.status !== 'ok') {
      anCount.textContent = res.status === 'loading' ? 'loading' : 'analyses.json could not be read' + (res.error ? ': ' + res.error : '');
      return;
    }
    const all = res.data;
    const num = (v) => (v === '' || v == null ? -1 : Number(v));
    const q = an.q.trim().toLowerCase();
    let rows = all.filter((r) => an.kind === 'all' || r.kind === an.kind)
      .filter((r) => !q || (r.area + ' ' + r.file).toLowerCase().includes(q));
    const by = {
      written: (a, b) => String(b.written_utc).localeCompare(String(a.written_utc)),
      oldest: (a, b) => String(a.written_utc).localeCompare(String(b.written_utc)),
      area: (a, b) => a.area.localeCompare(b.area) || a.file.localeCompare(b.file),
      rows: (a, b) => num(b.n_rows) - num(a.n_rows) || by.area(a, b),
      size: (a, b) => Number(b.size_kb) - Number(a.size_kb) || by.area(a, b),
    };
    rows = rows.slice().sort(by[an.sort]);
    anCount.textContent = Math.min(rows.length, SHOWN) + ' shown of ' + rows.length + ' matching, ' + all.length + ' files';
    const frag = document.createDocumentFragment();
    rows.slice(0, SHOWN).forEach((r) => {
      const tr = document.createElement('tr');
      tr.dataset.key = r.path;
      tr.appendChild(td(r.area, 'id'));
      tr.appendChild(td(r.file, 'id'));
      tr.appendChild(td(r.kind));
      tr.appendChild(td(r.n_rows === '' ? '—' : fmtInt(Number(r.n_rows)), 'num'));
      tr.appendChild(td(fmtInt(Number(r.size_kb)), 'num'));
      const d = parse(r.written_utc);
      tr.appendChild(td(d ? fmtDay(d) : '—', 'num'));
      const src = document.createElement('td');
      src.appendChild(pill(r.written_source, r.written_source === 'git' ? '' : 'warn'));
      tr.appendChild(src);
      tr.appendChild(td(r.tracked ? 'git' : 'local only'));
      frag.appendChild(tr);
    });
    anBody.appendChild(frag);
  }

  document.querySelectorAll('[data-an-kind]').forEach((btn) => {
    btn.addEventListener('click', () => {
      an.kind = btn.dataset.anKind;
      document.querySelectorAll('[data-an-kind]').forEach((x) => x.setAttribute('aria-pressed', String(x === btn)));
      drawAnalyses();
    });
  });
  document.getElementById('an-q').addEventListener('input', (e) => { an.q = e.target.value; drawAnalyses(); });
  document.getElementById('an-sort').addEventListener('change', (e) => { an.sort = e.target.value; drawAnalyses(); });

  document.querySelectorAll('[data-tab]').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-tab]').forEach((x) => x.setAttribute('aria-pressed', String(x === btn)));
      ['lanes', 'analyses'].forEach((t) => { document.getElementById('tab-' + t).hidden = btn.dataset.tab !== t; });
    });
  });

  dash.onData(() => { draw(); drawAnalyses(); });
})();
