// Light / dark theme. Loaded in <head>, without defer, so the theme is set before the first paint.
(function () {
  const root = document.documentElement;
  const KEY = 'er-theme';
  const system = window.matchMedia('(prefers-color-scheme: dark)');

  // storage can be blocked (private windows, some file:// setups): the theme then lasts for the page only
  function saved() {
    try {
      const value = localStorage.getItem(KEY);
      return value === 'light' || value === 'dark' ? value : null;
    } catch (e) {
      return null;
    }
  }

  function apply(theme) {
    root.dataset.theme = theme;
    const btn = document.querySelector('.theme-toggle');
    if (!btn) return;
    btn.title = theme === 'dark' ? 'Dark Theme' : 'Light Theme';
    btn.setAttribute('aria-label', 'Switch to ' + (theme === 'dark' ? 'light' : 'dark') + ' theme');
  }

  // The reader's own choice wins; without one, follow the system setting.
  apply(saved() || (system.matches ? 'dark' : 'light'));

  document.addEventListener('DOMContentLoaded', function () {
    apply(root.dataset.theme);                                   // the button exists now: label it
    const btn = document.querySelector('.theme-toggle');
    if (!btn) return;
    btn.addEventListener('click', function () {
      const next = root.dataset.theme === 'dark' ? 'light' : 'dark';
      try { localStorage.setItem(KEY, next); } catch (e) { /* see saved() */ }
      apply(next);
    });
  });

  system.addEventListener('change', function (e) {
    if (!saved()) apply(e.matches ? 'dark' : 'light');
  });
})();
