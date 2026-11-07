// Injected into the page. Numbers every visible interactive element with a
// data-mp-ref attribute and returns a compact, form-aware description of the
// page so the LLM can act on "[ref]" ids instead of raw HTML.
// page.evaluate passes ONE argument, so the limits arrive as an array.
([maxElements, maxText]) => {
  const SELECTOR = [
    'a[href]', 'button', 'input:not([type=hidden])', 'select', 'textarea', 'summary',
    '[role=button]', '[role=link]', '[role=checkbox]', '[role=radio]', '[role=tab]',
    '[role=menuitem]', '[role=option]', '[role=combobox]', '[role=switch]',
    '[contenteditable=true]',
  ].join(',');

  const clean = (s, n = 120) => (s || '').replace(/\s+/g, ' ').trim().slice(0, n);

  const isVisible = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) {
      // Visually hidden radios/checkboxes are often styled via their label.
      if (!(el.type === 'radio' || el.type === 'checkbox')) return false;
    }
    const st = getComputedStyle(el);
    if (st.display === 'none' || st.visibility === 'hidden') return false;
    if (el.closest('[hidden],[aria-hidden=true]') && !(el.type === 'radio' || el.type === 'checkbox')) return false;
    return true;
  };

  const textById = (ids) => (ids || '').split(/\s+/).map((id) => {
    const n = id && document.getElementById(id);
    return n ? n.innerText : '';
  }).join(' ');

  const accessibleName = (el) => {
    if (el.getAttribute('aria-label')) return el.getAttribute('aria-label');
    if (el.getAttribute('aria-labelledby')) return textById(el.getAttribute('aria-labelledby'));
    if (el.labels && el.labels.length) return Array.from(el.labels).map((l) => l.innerText).join(' ');
    const tag = el.tagName.toLowerCase();
    if (tag === 'input' || tag === 'textarea' || tag === 'select') {
      return el.placeholder || el.title || el.name || '';
    }
    return el.innerText || el.value || el.title || (el.querySelector('img') || {}).alt || '';
  };

  const fieldError = (el) => {
    const described = textById(el.getAttribute('aria-describedby'));
    const group = el.closest('.govuk-form-group--error, .nhsuk-form-group--error, [class*="error"]');
    const msg = group && group.querySelector('.govuk-error-message, .nhsuk-error-message, [class*="error-message"]');
    const t = clean((msg && msg.innerText) || '', 160);
    if (t) return t;
    return /error/i.test(described) ? clean(described, 160) : '';
  };

  const hint = (el) => {
    const d = clean(textById(el.getAttribute('aria-describedby')), 160);
    return d && !/error/i.test(d) ? d : '';
  };

  document.querySelectorAll('[data-mp-ref]').forEach((n) => n.removeAttribute('data-mp-ref'));

  const vh = window.innerHeight;
  const elements = [];
  let ref = 0;
  for (const el of document.querySelectorAll(SELECTOR)) {
    if (elements.length >= maxElements) break;
    if (!isVisible(el)) continue;
    // Footer links (terms, licences, sitemaps) cost tokens and are never the task.
    if (el.closest('footer, [role=contentinfo]')) continue;
    ref += 1;
    el.setAttribute('data-mp-ref', String(ref));
    const tag = el.tagName.toLowerCase();
    const role = el.getAttribute('role') || '';
    const type = (el.getAttribute('type') || '').toLowerCase();
    const r = el.getBoundingClientRect();
    const item = {
      ref,
      tag,
      role,
      type,
      name: clean(accessibleName(el)),
      in_view: r.bottom > 0 && r.top < vh,
    };
    if (tag === 'a') item.href = el.getAttribute('href');
    if (tag === 'input' || tag === 'textarea' || tag === 'select') {
      // The fieldset legend carries the real question for radios and for
      // split fields like Day / Month / Year ("Date of birth").
      const fs = el.closest('fieldset');
      const legend = fs && fs.querySelector('legend');
      if (legend) item.question = clean(legend.innerText, 160);
    }
    if (tag === 'input' || tag === 'textarea') {
      if (type === 'radio' || type === 'checkbox') {
        item.checked = el.checked;
        if (el.name) item.group = el.name;
      } else {
        item.value = el.value;
      }
      if (el.autocomplete && el.autocomplete !== 'on' && el.autocomplete !== 'off') item.autocomplete = el.autocomplete;
    }
    if (tag === 'select') {
      item.value = el.value;
      item.options = Array.from(el.options).slice(0, 40).map((o) => clean(o.text, 60));
    }
    if (el.required || el.getAttribute('aria-required') === 'true') item.required = true;
    if (el.disabled) item.disabled = true;
    const err = fieldError(el);
    if (err) item.error = err;
    const h = hint(el);
    if (h && (tag === 'input' || tag === 'select' || tag === 'textarea')) item.hint = h;
    elements.push(item);
  }

  const main = document.querySelector('main') || document.body;
  // Only real form-validation summaries; [role=alert] is also used for cookie
  // banners and "call 111" callouts, which are not errors.
  const errorSummary = Array.from(document.querySelectorAll(
    '.govuk-error-summary, .nhsuk-error-summary, .error-summary, [class*="error-summary"], [role=alert]'))
    .find((n) => isVisible(n) && !/cookie/i.test(n.innerText)
      && /(there is a problem|there was a problem|error|must |enter |select |choose )/i.test(n.innerText));
  return {
    url: location.href,
    title: document.title,
    headings: Array.from(main.querySelectorAll('h1, h2')).slice(0, 12).map((h) => clean(h.innerText, 100)),
    error_summary: errorSummary ? clean(errorSummary.innerText, 400) : '',
    text: clean(main.innerText, maxText),
    elements,
    scroll: { y: Math.round(window.scrollY), max: Math.round(document.documentElement.scrollHeight - vh) },
  };
}
