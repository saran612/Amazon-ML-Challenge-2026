// Documentation search. The index (search-index.js, written by scripts/build_site.py) holds one entry per page
// and per sub-heading. It is a script, not JSON, so it also loads from file://, and it is fetched on first use.
(function () {
  const box = document.querySelector('.search');
  if (!box) return;
  const input = box.querySelector('input');
  const panel = box.querySelector('.search-results');
  const root = document.currentScript.dataset.root || '';        // path from this page to the site root
  const LIMIT = 8;
  let index = null;
  let loading = false;
  let hits = [];
  let active = -1;

  function load() {
    if (index || loading) return;
    loading = true;
    const script = document.createElement('script');
    script.src = root + 'search-index.js';
    script.onload = function () {
      index = (window.SEARCH_INDEX || []).map(function (e) {
        return { entry: e, title: e.t.toLowerCase(), text: e.x.toLowerCase() };
      });
      run();
    };
    script.onerror = function () { loading = false; };
    document.head.appendChild(script);
  }

  function terms(query) {
    return query.toLowerCase().split(/\s+/).filter(Boolean);
  }

  // Every term must appear in the entry. A match in the title counts far more than one in the text.
  function search(words) {
    const found = [];
    index.forEach(function (item, order) {
      let score = 0;
      for (const word of words) {
        const inTitle = item.title.indexOf(word);
        const inText = item.text.indexOf(word);
        if (inTitle < 0 && inText < 0) return;
        if (inTitle >= 0) score += inTitle === 0 ? 12 : 8;
        if (inText >= 0) score += 1 + Math.min(item.text.split(word).length - 1, 5) * 0.5;
      }
      found.push({ entry: item.entry, text: item.text, score: score, order: order });
    });
    found.sort(function (a, b) { return b.score - a.score || a.order - b.order; });
    return found.slice(0, LIMIT);
  }

  // Text with the matched words wrapped in <mark>, built from nodes so page text is never parsed as HTML.
  function marked(text, words) {
    const out = document.createDocumentFragment();
    const pattern = new RegExp('(' + words.map(function (w) {
      return w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }).join('|') + ')', 'gi');
    text.split(pattern).forEach(function (part, i) {
      if (!part) return;
      if (i % 2) {
        const mark = document.createElement('mark');
        mark.textContent = part;
        out.appendChild(mark);
      } else {
        out.appendChild(document.createTextNode(part));
      }
    });
    return out;
  }

  // About 140 characters of the entry's text around the first matched word.
  function snippet(hit, words) {
    const text = hit.entry.x;
    let at = -1;
    for (const word of words) {
      const i = hit.text.indexOf(word);
      if (i >= 0 && (at < 0 || i < at)) at = i;
    }
    let start = Math.max(0, at - 50);
    if (start > 0) start = text.indexOf(' ', start) + 1 || start;      // begin on a whole word
    const end = Math.min(text.length, start + 140);
    return (start > 0 ? '… ' : '') + text.slice(start, end).trim() + (end < text.length ? ' …' : '');
  }

  function span(className, content) {
    const el = document.createElement('span');
    el.className = className;
    el.appendChild(typeof content === 'string' ? document.createTextNode(content) : content);
    return el;
  }

  function setActive(i, fromPointer) {
    const links = panel.querySelectorAll('.search-hit');
    links.forEach(function (link, n) {
      link.classList.toggle('active', n === i);
      link.setAttribute('aria-selected', String(n === i));
    });
    active = i;
    if (links[i] && !fromPointer) links[i].scrollIntoView({ block: 'nearest' });   // the pointer is already on it
  }

  function open(isOpen) {
    panel.hidden = !isOpen;
    input.setAttribute('aria-expanded', String(isOpen));
  }

  function run() {
    const words = terms(input.value);
    panel.textContent = '';
    hits = [];
    active = -1;
    if (!words.length || !index) { open(false); return; }

    hits = search(words);
    if (!hits.length) {
      const empty = document.createElement('div');
      empty.className = 'search-empty';
      empty.textContent = 'No results for "' + input.value.trim() + '"';
      panel.appendChild(empty);
    }
    hits.forEach(function (hit, n) {
      const link = document.createElement('a');
      link.className = 'search-hit';
      link.href = root + hit.entry.u;
      link.setAttribute('role', 'option');
      link.appendChild(span('hit-title', marked(hit.entry.t, words)));
      link.appendChild(span('hit-path', hit.entry.p));
      if (hit.entry.x) link.appendChild(span('hit-text', marked(snippet(hit, words), words)));
      link.addEventListener('click', function () { open(false); });   // a hit on this page only scrolls
      link.addEventListener('mousemove', function () { if (active !== n) setActive(n, true); });
      panel.appendChild(link);
    });
    open(true);
    if (hits.length) setActive(0);
  }

  input.addEventListener('focus', function () { load(); run(); });
  input.addEventListener('input', function () { load(); run(); });

  input.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      if (!hits.length) return;
      e.preventDefault();
      setActive((active + (e.key === 'ArrowDown' ? 1 : hits.length - 1)) % hits.length);
    } else if (e.key === 'Enter') {
      const link = panel.querySelectorAll('.search-hit')[Math.max(active, 0)];
      if (link) { e.preventDefault(); link.click(); input.blur(); }
    } else if (e.key === 'Escape') {
      open(false);
      input.blur();
    }
  });

  // "/" or Ctrl/Cmd+K jumps to the search box
  document.addEventListener('keydown', function (e) {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName) || e.target.isContentEditable;
    const slash = e.key === '/' && !typing && !e.ctrlKey && !e.metaKey && !e.altKey;
    const commandK = (e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k';
    if (!slash && !commandK) return;
    e.preventDefault();
    input.focus();
    input.select();
  });

  document.addEventListener('click', function (e) {
    if (!box.contains(e.target)) open(false);
  });
})();
