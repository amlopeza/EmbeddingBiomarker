"""Live edit-and-resync for the manuscript HTML.

The paper is a self-contained HTML opened directly in a browser. Browsers cannot
silently write back to disk from a file:// page, so the round-trip is:

    1. `inject`  — make the HTML editable: stamp every editable block with a
                   stable data-eid and embed a small editor (Edit / Save toolbar).
    2. user edits in the browser, clicks **Save** -> writes/downloads
       `<doc>.changes.json` (id-anchored diff of what was changed by hand).
    3. `apply`   — agent reads that JSON and rewrites the source HTML by data-eid,
                   so the file matches the manual edits; the agent then continues.

Stdlib only (no venv needed).

Usage:
    python scripts/paper_editor.py inject paper/embedbiomarker.html
    python scripts/paper_editor.py status paper/embedbiomarker.changes.json
    python scripts/paper_editor.py apply  paper/embedbiomarker.html [changes.json]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

EDITABLE = ["h1", "h2", "h3", "p", "li", "caption", "figcaption", "td", "th"]
OPEN_TAG = re.compile(r"<(?:" + "|".join(EDITABLE) + r")(?:\s[^>]*)?>", re.I)
RUNTIME_MARKER = 'id="pe-runtime"'

# --------------------------------------------------------------------------- #
# Editor runtime (embedded into the HTML by `inject`)
# --------------------------------------------------------------------------- #
PE_STYLE = """
<style id="pe-style">
  #pe-bar{position:fixed;right:16px;bottom:16px;z-index:9999;
    font-family:"Helvetica Neue",Arial,sans-serif;font-size:13px;
    background:#1a1a1a;color:#fff;border-radius:8px;padding:8px 10px;
    box-shadow:0 2px 10px rgba(0,0,0,.3);display:flex;gap:8px;align-items:center;}
  #pe-bar button{font:inherit;cursor:pointer;border:0;border-radius:5px;
    padding:5px 10px;background:#3a3a3a;color:#fff;}
  #pe-bar button.pe-on{background:#7a1f2b;}
  #pe-bar #pe-status{color:#bdbdbd;max-width:280px;}
  body.pe-editing [data-eid]{outline:1px dashed #c9a23a;outline-offset:2px;}
  body.pe-editing [data-eid]:focus{outline:2px solid #7a1f2b;background:#fffdf3;}
  @media print{#pe-bar{display:none!important;}
    body.pe-editing [data-eid]{outline:none!important;background:none!important;}}
</style>
"""

PE_RUNTIME = """
<script id="pe-runtime">
(function(){
  var DOC = "__DOC__";
  function els(){ return Array.prototype.slice.call(document.querySelectorAll('[data-eid]')); }
  var baseline = {};
  function capture(){ els().forEach(function(e){ baseline[e.dataset.eid] = e.innerHTML; }); }
  function sectionOf(el){
    var n = el;
    while(n){
      var p = n.previousElementSibling;
      while(p){ if(p.tagName === 'H2' || p.tagName === 'H3'){ return p.textContent.trim(); } p = p.previousElementSibling; }
      n = n.parentElement;
    }
    return "";
  }
  var editing = false, btnEdit, statusEl;
  function setStatus(t){ statusEl.textContent = t; }
  function setEditing(on){
    editing = on;
    els().forEach(function(e){ e.contentEditable = on ? "true" : "false"; });
    document.body.classList.toggle('pe-editing', on);
    btnEdit.textContent = on ? "Editing…" : "Edit";
    btnEdit.classList.toggle('pe-on', on);
    setStatus(on ? "click text to edit, then Save" : "");
  }
  function changes(){
    var out = [];
    els().forEach(function(e){
      var cur = e.innerHTML, id = e.dataset.eid;
      if(baseline[id] !== undefined && cur !== baseline[id]){
        out.push({eid:id, tag:e.tagName.toLowerCase(), section:sectionOf(e),
                  original:baseline[id], current:cur});
      }
    });
    return out;
  }
  function save(){
    var ch = changes();
    var payload = {doc:DOC, saved_at:new Date().toISOString(), n_changes:ch.length, changes:ch};
    var text = JSON.stringify(payload, null, 2);
    var name = DOC.replace(/\\.html$/, '') + '.changes.json';
    if(ch.length === 0){ setStatus("no manual changes to save"); return; }
    if(window.showSaveFilePicker){
      window.showSaveFilePicker({suggestedName:name,
        types:[{description:'JSON', accept:{'application/json':['.json']}}]})
        .then(function(h){ return h.createWritable(); })
        .then(function(w){ return w.write(text).then(function(){ return w.close(); }); })
        .then(function(){ setStatus("saved " + ch.length + " change(s) → " + name); })
        .catch(function(err){ if(err && err.name === 'AbortError'){ return; } download(text, name, ch.length); });
      return;
    }
    download(text, name, ch.length);
  }
  function download(text, name, n){
    var blob = new Blob([text], {type:'application/json'});
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = name; a.click();
    setStatus("downloaded " + name + " (" + n + ") — move next to the .html, then tell the agent");
  }
  function bar(){
    var b = document.createElement('div'); b.id = 'pe-bar';
    btnEdit = document.createElement('button'); btnEdit.textContent = 'Edit';
    btnEdit.onclick = function(){ setEditing(!editing); };
    var btnSave = document.createElement('button'); btnSave.textContent = 'Save';
    btnSave.onclick = save;
    statusEl = document.createElement('span'); statusEl.id = 'pe-status';
    b.appendChild(btnEdit); b.appendChild(btnSave); b.appendChild(statusEl);
    document.body.appendChild(b);
  }
  if(document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', function(){ capture(); bar(); });
  } else { capture(); bar(); }
})();
</script>
"""


def inject(html_path: Path) -> None:
    html = html_path.read_text(encoding="utf-8")
    n_existing = html.count("data-eid=")
    counter = [0]

    def add_eid(m: re.Match) -> str:
        full = m.group(0)
        if "data-eid=" in full:
            return full
        counter[0] += 1
        return full[:-1] + f' data-eid="e{counter[0]:04d}">'

    # only stamp tags that don't already carry an id
    html = OPEN_TAG.sub(add_eid, html)

    if RUNTIME_MARKER not in html:
        runtime = PE_RUNTIME.replace("__DOC__", html_path.name)
        html = html.replace("</body>", PE_STYLE + runtime + "\n</body>", 1)
        injected_runtime = True
    else:
        injected_runtime = False

    html_path.write_text(html, encoding="utf-8")
    print(f"inject: +{counter[0]} new data-eid (was {n_existing}); "
          f"runtime {'added' if injected_runtime else 'already present'} -> {html_path}")


def status(changes_path: Path) -> None:
    data = json.loads(changes_path.read_text(encoding="utf-8"))
    print(f"{data.get('doc')}  saved_at={data.get('saved_at')}  "
          f"changes={data.get('n_changes')}")
    for c in data.get("changes", []):
        orig = re.sub(r"<[^>]+>", "", c["original"]).strip()
        cur = re.sub(r"<[^>]+>", "", c["current"]).strip()
        print(f"\n  [{c['eid']} <{c['tag']}> | {c['section']}]")
        print(f"    - {orig[:160]}")
        print(f"    + {cur[:160]}")


def apply(html_path: Path, changes_path: Path | None) -> None:
    if changes_path is None:
        changes_path = html_path.with_suffix("").with_suffix(".changes.json")
        # paper/embedbiomarker.html -> paper/embedbiomarker.changes.json
        changes_path = html_path.parent / (html_path.stem + ".changes.json")
    data = json.loads(Path(changes_path).read_text(encoding="utf-8"))
    html = html_path.read_text(encoding="utf-8")
    applied, missed = 0, []
    for c in data.get("changes", []):
        eid, tag, new = c["eid"], c["tag"], c["current"]
        pat = re.compile(
            r'(<' + tag + r'\b[^>]*\bdata-eid="' + re.escape(eid) + r'"[^>]*>)(.*?)(</' + tag + r'>)',
            re.S | re.I,
        )
        html, n = pat.subn(lambda m: m.group(1) + new + m.group(3), html, count=1)
        if n:
            applied += 1
        else:
            missed.append(eid)
    html_path.write_text(html, encoding="utf-8")
    print(f"apply: {applied}/{len(data.get('changes', []))} change(s) -> {html_path}")
    if missed:
        print(f"  WARNING: no element matched for eid(s): {', '.join(missed)}")
    archive = Path(changes_path).with_suffix(".applied.json")
    Path(changes_path).rename(archive)
    print(f"  archived {changes_path} -> {archive}")


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 1
    cmd, target = argv[1], Path(argv[2])
    if cmd == "inject":
        inject(target)
    elif cmd == "status":
        status(target)
    elif cmd == "apply":
        apply(target, Path(argv[3]) if len(argv) > 3 else None)
    else:
        print(f"unknown command {cmd!r}")
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
