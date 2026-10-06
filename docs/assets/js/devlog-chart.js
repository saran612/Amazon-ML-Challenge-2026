// Development-log chart: macro F0.5 per step. The table under it holds the same values.
(function () {
  const host = document.querySelector('[data-chart="devlog"]');
  if (!host) return;
  const fig = host.closest('.fig');
  const NS = 'http://www.w3.org/2000/svg';
  const data = [
    { step: 'Step 1', name: 'Full pipeline', f: 0.9775, p: 0.994, r: 0.949, kept: true, label: true },
    { step: 'Step 2', name: 'More candidates', f: 0.9777, p: 0.994, r: 0.949, kept: true },
    { step: 'Step 3', name: 'New features', f: 0.9830, p: 0.995, r: 0.963, kept: true },
    { step: 'Step 4', name: 'Cross-encoder', f: 0.9861, p: 0.998, r: 0.964, kept: true, label: true, note: 'default' },
    { step: 'Step 5', name: 'Denser crowd', f: 0.9850, p: 0.996, r: 0.965, kept: false, label: true, note: 'optional' }
  ];
  const W = 690, H = 268, L = 54, R = 20, T = 30, B = 58;
  const lo = 0.975, hi = 0.996, ceiling = 0.995;
  const bottom = H - B;
  const x = function (i) { return L + (i + 0.5) * (W - L - R) / data.length; };
  const y = function (v) { return T + (hi - v) / (hi - lo) * (bottom - T); };

  function el(name, attrs, text) {
    const node = document.createElementNS(NS, name);
    Object.keys(attrs || {}).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    if (text !== undefined) node.textContent = text;
    return node;
  }

  const svg = el('svg', {
    viewBox: '0 0 ' + W + ' ' + H, role: 'img',
    'aria-label': 'Macro F0.5 rises from 0.9775 at step 1 to 0.9861 at step 4; the optional step 5 scores 0.9850.'
  });

  [0.975, 0.980, 0.985, 0.990].forEach(function (t) {
    svg.appendChild(el('line', { class: 'd-grid', x1: L, x2: W - R, y1: y(t), y2: y(t) }));
    svg.appendChild(el('text', { class: 'd-tick', x: L - 10, y: y(t) + 4, 'text-anchor': 'end' }, t.toFixed(3)));
  });
  svg.appendChild(el('line', { class: 'd-axis', x1: L, x2: W - R, y1: y(ceiling), y2: y(ceiling) }));
  svg.appendChild(el('text', { class: 'd-tick', x: L - 10, y: y(ceiling) + 4, 'text-anchor': 'end' }, ceiling.toFixed(3)));
  svg.appendChild(el('text', { class: 'd-s', x: L + 6, y: y(ceiling) - 8 },
    'Ceiling: a perfect matcher on the blocking candidates'));

  const kept = data.map(function (d, i) { return { d: d, i: i }; }).filter(function (o) { return o.d.kept; });
  const last = kept[kept.length - 1];
  data.forEach(function (d, i) {           // branches that were tried but not adopted
    if (!d.kept) svg.appendChild(el('path', {
      class: 'c-line alt', d: 'M' + x(last.i) + ',' + y(last.d.f) + ' L' + x(i) + ',' + y(d.f)
    }));
  });
  svg.appendChild(el('path', {
    class: 'c-line',
    d: kept.map(function (o, n) { return (n ? 'L' : 'M') + x(o.i) + ',' + y(o.d.f); }).join(' ')
  }));

  const tip = document.createElement('div');
  tip.className = 'tip';
  fig.appendChild(tip);

  function show(d, target) {
    tip.textContent = '';
    const value = document.createElement('b');
    value.textContent = d.f.toFixed(4);
    const name = document.createElement('span');
    name.textContent = d.step + ': ' + d.name;
    const detail = document.createElement('span');
    detail.textContent = 'precision ' + d.p.toFixed(3) + ' · recall ' + d.r.toFixed(3);
    tip.append(value, name, document.createElement('br'), detail);
    tip.style.display = 'block';
    const box = target.getBoundingClientRect(), frame = fig.getBoundingClientRect();
    const half = tip.offsetWidth / 2 + 8;
    const left = box.left + box.width / 2 - frame.left;
    tip.style.left = Math.min(Math.max(left, half), frame.width - half) + 'px';
    tip.style.top = (box.top - frame.top - 4) + 'px';
  }
  function hide() { tip.style.display = 'none'; }

  data.forEach(function (d, i) {
    svg.appendChild(el('text', { class: 'd-tick', x: x(i), y: bottom + 22, 'text-anchor': 'middle' }, d.step));
    svg.appendChild(el('text', { x: x(i), y: bottom + 39, 'text-anchor': 'middle' }, d.name));
    svg.appendChild(el('circle', { class: d.kept ? 'c-dot' : 'c-dot open', cx: x(i), cy: y(d.f), r: 5 }));
    if (d.label) {
      svg.appendChild(el('text', { class: 'd-t', x: x(i), y: y(d.f) - 14, 'text-anchor': 'middle' },
        d.f.toFixed(4) + (d.note ? ' · ' + d.note : '')));
    }
    const hit = el('circle', {
      class: 'c-hit', cx: x(i), cy: y(d.f), r: 16, tabindex: 0,
      'aria-label': d.step + ', ' + d.name + ': macro F0.5 ' + d.f.toFixed(4) +
        ', precision ' + d.p.toFixed(3) + ', recall ' + d.r.toFixed(3)
    });
    hit.addEventListener('pointerenter', function () { show(d, hit); });
    hit.addEventListener('pointerleave', hide);
    hit.addEventListener('focus', function () { show(d, hit); });
    hit.addEventListener('blur', hide);
    svg.appendChild(hit);
  });

  host.appendChild(svg);
})();
