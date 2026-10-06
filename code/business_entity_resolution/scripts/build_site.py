#!/usr/bin/env python3
"""Build the documentation site: web-src/ -> docs/ (standard library only).

Each file under web-src/pages/ holds the content of one page and nothing else. This script wraps it in
web-src/layout.html, adds the sidebar (page order from web-src/nav.json, sub-items from the page's
<h3 id="..."> headings) and the previous/next links, and writes the result to the same path under docs/.
docs/assets/ is edited in place and is not touched. docs/ is the folder GitHub Pages serves, so
docs/index.html is the site's starting page; the Markdown files and images already in docs/ are left alone.

It also writes docs/search-index.js, the text of every page and sub-heading, which the search box in the
top bar (docs/assets/js/search.js) loads the first time it is used.

A page marked "standalone": true in nav.json is not part of the documentation: it is left out of the
sidebar and the previous/next chain, and is built without the layout's {{#docs}} ... {{/docs}} blocks
(sidebar, previous/next links, footer), so it shows the top bar and its own content only. It gets the
{{#standalone}} ... {{/standalone}} blocks instead: in the top bar, a Documentation link back to the home
page takes the place of the Dataset link that leads to it.

Usage:
    python scripts/build_site.py            # write docs/**/*.html
    python scripts/build_site.py --check    # write nothing; fail if docs/ is out of date (used in CI)

The build stops with an error when a page and nav.json disagree, when an id is used twice on a page,
or when a local link points at a page, heading or file that does not exist.
"""
import argparse
import html
import json
import posixpath
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "web-src"
OUT = ROOT / "docs"
SITE = "Entity Resolver"
HOME = "index.html"
EXTERNAL = ("http://", "https://", "mailto:", "data:", "//")

SECTION = re.compile(r'<section id="([^"]+)"')
H3 = re.compile(r'<h3 id="([^"]+)">(.*?)</h3>', re.S)
ID = re.compile(r'(?<![\w-])id="([^"]+)"')
REF = re.compile(r'(?<![\w-])(?:href|src)="([^"]*)"')
TAG = re.compile(r"<[^>]+>")
BLOCK = re.compile(r"^\{\{#(docs|standalone)\}\}\n(.*?)^\{\{/\1\}\}\n", re.S | re.M)   # layout parts for one kind of page
SEARCH_INDEX = "search-index.js"
# left out of the search text: scripts, drawings, and the eyebrow and title that open every page
UNSEARCHED = re.compile(r'<(script|svg)\b.*?</\1>|<p class="eyebrow">.*?</p>|<h[12]>.*?</h[12]>', re.S)

# arrow icons beside the "Previous" and "Next" labels, drawn like the other stroke icons in the layout
ICON = ('<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{}</svg>')
ARROW_LEFT = ICON.format('<line x1="19" y1="12" x2="5" y2="12"></line><polyline points="12 19 5 12 12 5"></polyline>')
ARROW_RIGHT = ICON.format('<line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline>')


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def rel(target: str, page: dict) -> str:
    """Relative URL of a site path, as seen from a page. The site is hosted under a sub-path, so no link starts with /."""
    return posixpath.relpath(target, posixpath.dirname(page["path"]) or ".")


def load(errors: list) -> list:
    groups = json.loads((SRC / "nav.json").read_text(encoding="utf-8"))
    for group in groups:
        for page in group["pages"]:
            source = SRC / "pages" / page["path"]
            if not source.is_file():
                errors.append(f"nav.json lists {page['path']}, but web-src/pages/{page['path']} does not exist")
                page["body"] = ""
            else:
                page["body"] = source.read_text(encoding="utf-8").strip("\n")
            page["sections"] = [(sid, " ".join(TAG.sub("", title).split())) for sid, title in H3.findall(page["body"])]
    listed = {page["path"] for group in groups for page in group["pages"]}
    for source in sorted((SRC / "pages").rglob("*.html")):
        path = source.relative_to(SRC / "pages").as_posix()
        if path not in listed:
            errors.append(f"web-src/pages/{path} is not listed in nav.json")
    return groups


def sidebar(page: dict, groups: list) -> str:
    out = []
    for group in groups:
        out += ['        <div class="nav-group">', f'          <h4>{esc(group["title"])}</h4>', '          <ul class="nav-list">']
        for p in group["pages"]:
            if p.get("standalone"):
                continue
            here = p is page
            href = rel(p["path"], page)
            current = ' aria-current="page"' if here else ""
            link = f'<a href="{href}"{current}>{esc(p["title"])}</a>'
            if not p["sections"]:
                out.append(f'            <li class="has-active">{link}</li>' if here else f"            <li>{link}</li>")
                continue
            # the current page's sub-items start open, so they are reachable without JavaScript
            out += ['            <li class="has-active open">' if here else "            <li>", f"              {link}",
                    '              <ul class="nav-sub">']
            for sid, title in p["sections"]:
                out.append(f'                <li><a href="{"" if here else href}#{sid}">{title}</a></li>')
            out += ["              </ul>", "            </li>"]
        out += ["          </ul>", "        </div>", ""]
    return "\n".join(out)


def pager(page: dict, pages: list) -> str:
    if page.get("standalone"):
        return ""
    pages = [p for p in pages if not p.get("standalone")]
    i = pages.index(page)
    out = ['      <div class="pager">']
    if i > 0:
        p = pages[i - 1]
        label = f"<small>{ARROW_LEFT}Previous</small>"
        out.append(f'        <a class="prev" href="{rel(p["path"], page)}">{label}{esc(p["title"])}</a>')
    if i + 1 < len(pages):
        p = pages[i + 1]
        label = f"<small>Next{ARROW_RIGHT}</small>"
        out.append(f'        <a class="next" href="{rel(p["path"], page)}">{label}{esc(p["title"])}</a>')
    return "\n".join(out + ["      </div>"])


def redirects(pages: list) -> str:
    """Script for the home page: the docs used to be one page, so index.html#blocking must still find its section."""
    moved = {}
    for p in pages:
        if p["path"] == HOME:
            continue
        for sid in SECTION.findall(p["body"]):
            moved[sid] = p["path"]
        for sid, _ in p["sections"]:
            moved[sid] = f'{p["path"]}#{sid}'
    return ("  <script>\n"
            "    // Links into the old single-page docs (index.html#blocking) go to the page that now holds the section.\n"
            "    (function () {\n"
            f"      var moved = {json.dumps(moved, separators=(',', ':'))};\n"
            "      var to = moved[location.hash.slice(1)];\n"
            "      if (to) location.replace(to);\n"
            "    })();\n"
            "  </script>\n")


def render(page: dict, groups: list, pages: list, layout: str) -> str:
    home = page["path"] == HOME
    root = rel(".", page)
    kind = "standalone" if page.get("standalone") else "docs"
    layout = BLOCK.sub(lambda m: m.group(2) if m.group(1) == kind else "", layout)
    fields = {
        "source": f"pages/{page['path']}",
        "title": SITE if home else f"{esc(page['title'])} · {SITE}",
        "root": "" if root == "." else root + "/",
        "head": redirects(pages) if home else "",
        "sidebar": sidebar(page, groups),
        "pager": pager(page, pages),
        "content": page["body"],                       # last, so page text is never scanned for placeholders
    }
    for key, value in fields.items():
        layout = layout.replace("{{" + key + "}}", value)
    return layout


def plain(fragment: str) -> str:
    """The readable text of a piece of a page, on one line."""
    text = html.unescape(TAG.sub(" ", UNSEARCHED.sub(" ", fragment)))
    return re.sub(r" ([,.;:)])", r"\1", " ".join(text.split()))


def search_index(groups: list) -> str:
    """Data for assets/js/search.js: one entry per page (its text up to the first sub-heading) and per sub-heading."""
    entries = []
    for group in groups:
        for page in group["pages"]:
            body, marks = page["body"], list(H3.finditer(page["body"]))
            intro = body[:marks[0].start()] if marks else body
            entries.append({"t": page["title"], "p": group["title"], "u": page["path"], "x": plain(intro)})
            for (sid, title), mark, nxt in zip(page["sections"], marks, marks[1:] + [None]):
                text = plain(body[mark.end():nxt.start() if nxt else len(body)])
                crumbs = f'{group["title"]} › {page["title"]}'
                entries.append({"t": html.unescape(title), "p": crumbs, "u": f'{page["path"]}#{sid}', "x": text})
    rows = ",\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries)
    return f"// Generated by scripts/build_site.py from web-src/pages/. Do not edit.\nwindow.SEARCH_INDEX = [\n{rows}\n];\n"


def check_links(site: dict, errors: list) -> None:
    ids = {path: ID.findall(text) for path, text in site.items()}
    for path, text in site.items():
        for dup in sorted({i for i in ids[path] if ids[path].count(i) > 1}):
            errors.append(f'{path}: id="{dup}" is used more than once')
        for ref in REF.findall(text):
            if ref.startswith(EXTERNAL):
                continue
            target, _, anchor = ref.partition("#")
            target = posixpath.normpath(posixpath.join(posixpath.dirname(path), target)) if target else path
            if target in site:
                if anchor and anchor not in ids[target]:
                    errors.append(f'{path}: link "{ref}" points at a heading that does not exist')
            elif target.startswith("..") or not (OUT / target).is_file():
                errors.append(f'{path}: link "{ref}" points at a file that does not exist')


def build() -> tuple[dict, list]:
    errors: list = []
    groups = load(errors)
    pages = [page for group in groups for page in group["pages"]]
    layout = (SRC / "layout.html").read_text(encoding="utf-8")
    site = {page["path"]: render(page, groups, pages, layout) for page in pages}
    check_links(site, errors)
    for stray in sorted(OUT.rglob("*.html")):
        path = stray.relative_to(OUT).as_posix()
        if path not in site:
            errors.append(f"docs/{path} has no source in web-src/pages/ (delete it, or add the page there)")
    site[SEARCH_INDEX] = search_index(groups)          # added after the link check: it is data, not a page
    return site, errors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="write nothing; exit 1 if docs/ differs from a fresh build")
    args = ap.parse_args()

    site, errors = build()
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        sys.exit(1)

    stale = [path for path, text in site.items()
             if not (OUT / path).is_file() or (OUT / path).read_text(encoding="utf-8") != text]
    if args.check:
        if stale:
            for path in stale:
                print(f"OUT OF DATE: docs/{path}")
            print("run: python scripts/build_site.py")
            sys.exit(1)
        print(f"docs/ is up to date ({len(site) - 1} pages and the search index)")
        return
    for path in stale:
        (OUT / path).parent.mkdir(parents=True, exist_ok=True)
        with open(OUT / path, "w", encoding="utf-8", newline="\n") as f:
            f.write(site[path])
    print(f"built {len(site) - 1} pages and the search index into docs/ ({len(stale)} files changed)")


if __name__ == "__main__":
    main()
