(function () {
  const sidebar = document.querySelector('.sidebar-sticky');
  if (!sidebar) return;
  const pane = document.querySelector('.content');               // scrolls the documentation
  const rail = sidebar.closest('.sidebar') || sidebar;           // scrolls the navigation
  const chevron = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
    'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 6 15 12 9 18"></polyline></svg>';

  // Add an expand/collapse button to every item that has sub-items.
  sidebar.querySelectorAll('.nav-sub').forEach(function (sub) {
    const li = sub.parentElement;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'nav-toggle';
    btn.innerHTML = chevron;
    btn.setAttribute('aria-label', 'Toggle ' + li.querySelector('a').textContent.trim());
    btn.setAttribute('aria-expanded', String(li.classList.contains('open')));
    btn.addEventListener('click', function () { setOpen(li, !li.classList.contains('open')); });
    li.classList.add('has-sub');
    li.insertBefore(btn, sub);
  });

  function setOpen(li, open) {
    li.classList.toggle('open', open);
    const btn = li.querySelector(':scope > .nav-toggle');
    if (btn) btn.setAttribute('aria-expanded', String(open));
  }

  // keep a link visible inside the sidebar without scrolling the page
  function reveal(link) {
    const top = link.getBoundingClientRect().top - rail.getBoundingClientRect().top + rail.scrollTop;
    if (top < rail.scrollTop || top > rail.scrollTop + rail.clientHeight - 40) {
      rail.scrollTop = top - rail.clientHeight / 3;
    }
  }

  // Below 900px the sidebar is a drawer (see layout.css), opened by the menu button in the top bar.
  const menu = document.querySelector('.menu-toggle');
  const backdrop = document.createElement('div');
  backdrop.className = 'nav-backdrop';
  document.body.appendChild(backdrop);

  function setDrawer(open) {
    document.documentElement.classList.toggle('nav-open', open);
    if (!menu) return;
    menu.setAttribute('aria-expanded', String(open));
    menu.setAttribute('aria-label', (open ? 'Close' : 'Open') + ' the navigation');
  }

  if (menu) {
    menu.addEventListener('click', function () {
      setDrawer(!document.documentElement.classList.contains('nav-open'));
    });
  }
  backdrop.addEventListener('click', function () { setDrawer(false); });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') setDrawer(false); });
  sidebar.addEventListener('click', function (e) { if (e.target.closest('a')) setDrawer(false); });

  // The build marks the item of the page being shown. Its sub-links point at headings on this page.
  const page = sidebar.querySelector('li.has-active');
  if (!page) return;
  const links = Array.from(page.querySelectorAll('.nav-sub a'));
  let active = null;

  function setActive(link) {
    if (link === active) return;
    if (active) active.classList.remove('active');
    active = link;
    if (!link) return;
    link.classList.add('active');
    reveal(link);
  }

  links.forEach(function (link) {
    link.addEventListener('click', function () { setActive(link); });
  });

  // Scroll-spy: the active link is the last heading that has passed the top of the content pane.
  const targets = links
    .map(function (link) { return { link: link, el: document.getElementById(link.hash.slice(1)) }; })
    .filter(function (t) { return t.el; });

  function onScroll() {
    const edge = (pane ? pane.getBoundingClientRect().top : 0) + 60;
    let current = null;
    targets.forEach(function (t) {
      if (t.el.getBoundingClientRect().top <= edge) current = t.link;
    });
    setActive(current);
  }

  let ticking = false;
  function queueScroll() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(function () { onScroll(); ticking = false; });
  }
  window.addEventListener('scroll', queueScroll, { passive: true });
  if (pane) pane.addEventListener('scroll', queueScroll, { passive: true });

  reveal(page.querySelector('a'));
  onScroll();
})();
