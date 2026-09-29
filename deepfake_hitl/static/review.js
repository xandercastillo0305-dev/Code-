// Review interface: logged full-image reveal + client-side rationale check.
(function () {
  document.querySelectorAll('.reveal-btn').forEach(function (btn) {
    btn.addEventListener('click', async function () {
      var which = btn.dataset.which;
      if (!confirm('Reveal the full, unredacted ' + which + ' image? This action is recorded in the audit log.')) return;
      btn.disabled = true;
      var resp = await fetch(btn.dataset.url, { method: 'POST', headers: { 'X-CSRF-Token': window.CSRF_TOKEN } });
      if (!resp.ok) { alert('Reveal failed (' + resp.status + ').'); btn.disabled = false; return; }
      var url = URL.createObjectURL(await resp.blob());
      document.getElementById('full-' + which).src = url;
      document.getElementById('tag-' + which).textContent = 'REVEALED (logged)';
      btn.textContent = 'Revealed';
    });
  });

  var form = document.getElementById('review-form');
  if (!form) return;
  var ta = document.getElementById('rationale');
  var count = document.getElementById('rationale-count');
  var err = document.getElementById('rationale-error');
  var min = window.MIN_RATIONALE;
  function update() { count.textContent = ta.value.trim().length + ' / ' + min; }
  ta.addEventListener('input', update); update();
  form.addEventListener('submit', function (e) {
    err.textContent = '';
    if (!form.querySelector('input[name=decision]:checked')) {
      e.preventDefault(); err.textContent = 'Choose Confirm, Override or Flag.'; return;
    }
    if (ta.value.trim().length < min) {
      e.preventDefault(); err.textContent = 'Rationale must be at least ' + min + ' characters.'; ta.focus();
    }
  });
})();
