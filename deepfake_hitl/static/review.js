// Review interface: logged full-image reveal, overlay comparison, decision cards,
// rationale helpers and client-side validation (the server re-checks everything).
(function () {
  // --- logged full-image reveal (POST so it cannot be prefetched) ---
  document.querySelectorAll('.reveal-btn').forEach(function (btn) {
    btn.addEventListener('click', async function () {
      var which = btn.dataset.which;
      if (!confirm('Reveal the full, unredacted ' + which + ' image? This action is recorded in the audit log.')) return;
      btn.disabled = true;
      var resp = await fetch(btn.dataset.url, { method: 'POST', headers: { 'X-CSRF-Token': window.CSRF_TOKEN } });
      if (!resp.ok) { alert('Reveal failed (' + resp.status + ').'); btn.disabled = false; return; }
      var url = URL.createObjectURL(await resp.blob());
      document.getElementById('full-' + which).src = url;
      var tag = document.getElementById('tag-' + which);
      tag.textContent = 'REVEALED · logged';
      tag.classList.add('revealed');
      btn.textContent = 'Revealed (logged)';
    });
  });

  // --- overlay comparison slider ---
  var range = document.getElementById('overlay-range');
  if (range) {
    var top = document.getElementById('overlay-suspect');
    var out = document.getElementById('overlay-value');
    range.addEventListener('input', function () {
      top.style.opacity = range.value / 100;
      out.textContent = range.value + '%';
    });
  }

  // --- copy SHA-256 ---
  document.querySelectorAll('.copy-btn').forEach(function (b) {
    b.addEventListener('click', function () {
      navigator.clipboard && navigator.clipboard.writeText(b.dataset.copy).then(function () {
        b.textContent = 'Copied';
        setTimeout(function () { b.textContent = 'Copy'; }, 1500);
      });
    });
  });

  var form = document.getElementById('review-form');
  if (!form) return;

  // --- decision cards ---
  var cards = form.querySelectorAll('.decision-card');
  function markSelected() {
    cards.forEach(function (c) { c.classList.toggle('selected', c.querySelector('input').checked); });
  }
  form.querySelectorAll('input[name=decision]').forEach(function (r) { r.addEventListener('change', markSelected); });
  markSelected();

  // --- rationale: counter + quick-insert chips ---
  var ta = document.getElementById('rationale');
  var count = document.getElementById('rationale-count');
  var err = document.getElementById('rationale-error');
  var min = window.MIN_RATIONALE;
  function update() {
    var n = ta.value.trim().length;
    count.textContent = n + ' / ' + min;
    count.classList.toggle('text-success', n >= min);
    count.classList.toggle('text-muted', n < min);
  }
  ta.addEventListener('input', update); update();
  form.querySelectorAll('.chip').forEach(function (chip) {
    chip.addEventListener('click', function () {
      var text = chip.dataset.chip;
      var cur = ta.value.trim();
      ta.value = cur ? cur.replace(/[.;,]?$/, '; ') + text.charAt(0).toLowerCase() + text.slice(1) : text;
      ta.focus(); update();
    });
  });

  // --- validation + final confirmation ---
  form.addEventListener('submit', function (e) {
    err.textContent = '';
    var chosen = form.querySelector('input[name=decision]:checked');
    if (!chosen) { e.preventDefault(); err.textContent = 'Choose Confirm, Override or Flag.'; return; }
    if (ta.value.trim().length < min) {
      e.preventDefault(); err.textContent = 'Rationale must be at least ' + min + ' characters.'; ta.focus(); return;
    }
    var result = chosen.closest('.decision-card').querySelector('.decision-result b').textContent;
    if (!confirm('Submit "' + chosen.value + '"? Final classification: ' + result + '.\nThe case will become read-only.')) {
      e.preventDefault();
    }
  });
})();
