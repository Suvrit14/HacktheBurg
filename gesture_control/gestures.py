"""One Playwright function per gesture.

Every function takes the Playwright page and returns a short sentence saying
what happened, so the caller can print or log it.

    scroll_up    scroll the page up by 80% of the screen
    scroll_down  scroll the page down by 80% of the screen
    back         go to the previous page (browser back button)
    tab          move the highlight to the next link, button, field or option
                 (visits every radio, unlike the browser's Tab key)
    click        click the element that `tab` highlighted

Use apply_gesture(page, "back") to run a gesture by name.
"""

from playwright.sync_api import Page
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

SCROLL_FRACTION = 0.8  # how much of the screen one scroll moves
SCROLL_MAX_MS = 1500  # longest we wait for the smooth scroll to finish
CLICK_TIMEOUT_MS = 5000
NAV_TIMEOUT_MS = 10000

# The highlight is our own marker, not the CSS :focus state. :focus disappears
# as soon as the Chrome window stops being the active window (for example when
# you type the next command in the terminal), which made the outline blink.
#   data-mp-target     the element `click` will press
#   data-mp-highlight  the region drawn around it (for a radio or checkbox:
#                      the whole option, circle/box plus its label)
# !important beats sites that hide outlines.
HIGHLIGHT_CSS = """
[data-mp-highlight] {
  outline: 4px solid #ff8c00 !important;
  outline-offset: 3px !important;
  box-shadow: 0 0 0 10px rgba(255, 140, 0, 0.30) !important;
  background-color: rgba(255, 221, 0, 0.25) !important;
  border-radius: 2px;
}
"""

_ADD_HIGHLIGHT = """css => {
  if (document.getElementById('mp-focus-style')) return;
  const style = document.createElement('style');
  style.id = 'mp-focus-style';
  style.textContent = css;
  document.head.appendChild(style);
}"""

# The element the last `tab` picked (falls back to the focused element).
_TARGET = """(document.querySelector('[data-mp-target]') || document.activeElement)"""

_DESCRIBE_FOCUSED = """() => {
  const el = """ + _TARGET + """;
  if (!el || el === document.body || el === document.documentElement) return null;
  const label = el.labels && el.labels[0] ? el.labels[0].innerText : '';
  const img = el.querySelector && el.querySelector('img[alt]');
  const text = (el.getAttribute('aria-label') || el.innerText || label
                || el.getAttribute('title') || el.getAttribute('placeholder')
                || (img && img.alt) || el.getAttribute('name') || el.value
                || el.getAttribute('href') || '')
    .trim().replace(/\\s+/g, ' ').slice(0, 80);
  const tag = el.tagName.toLowerCase();
  const kind = (tag === 'input' && ['radio', 'checkbox'].includes(el.type))
    ? ` ${el.type}${el.checked ? ', selected' : ''}` : '';
  return `<${tag}${kind}> "${text}"`;
}"""

# Our own Tab: move focus to the next element that can be used, in page order.
# The browser's Tab key skips the other options in a radio group (it expects
# arrow keys), so with only tab + click you could pick just one radio. This
# walk visits every radio. Hidden things (collapsed conditional questions,
# display:none, disabled) are skipped. If focus is on something that isn't in
# the list (a page heading, <main> after "Skip to main content"), it moves to
# the next element after it.
# Elements that are only 1px big (visually hidden, like "Skip to main content"
# when it isn't focused) are skipped, because the highlight would be invisible.
# A radio or checkbox that is hidden this way still counts if its label shows.
_FOCUS_NEXT = """() => {
  const selector = 'a[href], button, input:not([type=hidden]), textarea, select, summary, '
                 + '[tabindex]:not([tabindex="-1"])';
  const shows = el => {
    if (!el || el.closest('[hidden], [inert]')) return false;
    if (getComputedStyle(el).visibility === 'hidden') return false;
    const r = el.getBoundingClientRect();
    return r.width > 1 && r.height > 1;
  };
  const isChoice = el => el.tagName === 'INPUT' && ['radio', 'checkbox'].includes(el.type);
  const usable = el => !el.disabled
    && (shows(el) || (isChoice(el) && el.labels && shows(el.labels[0])));
  const items = [...document.querySelectorAll(selector)].filter(usable);
  if (items.length === 0) return false;

  // Region to draw: for a radio/checkbox, the smallest box holding it and its label.
  const regionFor = el => {
    const label = isChoice(el) && el.labels && el.labels[0];
    if (!label) return el;
    let box = el.parentElement;
    while (box && !box.contains(label)) box = box.parentElement;
    return box || el;
  };

  const current = document.querySelector('[data-mp-target]') || document.activeElement;
  const index = items.indexOf(current);
  let next;
  if (index >= 0) {
    next = items[(index + 1) % items.length];
  } else if (current && current !== document.body) {
    next = items.find(el => current.compareDocumentPosition(el) & Node.DOCUMENT_POSITION_FOLLOWING);
  }
  next = next || items[0];

  document.querySelectorAll('[data-mp-target]').forEach(e => e.removeAttribute('data-mp-target'));
  document.querySelectorAll('[data-mp-highlight]').forEach(e => e.removeAttribute('data-mp-highlight'));
  next.setAttribute('data-mp-target', '');
  regionFor(next).setAttribute('data-mp-highlight', '');

  next.focus({ preventScroll: true });  // so typing (voice/LLM later) goes here
  next.scrollIntoView({ block: 'center', behavior: 'smooth' });
  return true;
}"""


# Smooth-scroll, then resolve once the animation ends ("scrollend"), with a
# timeout as a safety net. Resolves at once if the page can't move that way.
_SMOOTH_SCROLL = """([d, f, maxMs]) => new Promise(resolve => {
  const max = document.documentElement.scrollHeight - window.innerHeight;
  if ((d < 0 && window.scrollY <= 0) || (d > 0 && window.scrollY >= max - 1)) {
    resolve();
    return;
  }
  const timer = setTimeout(resolve, maxMs);
  window.addEventListener('scrollend', () => { clearTimeout(timer); resolve(); }, {once: true});
  window.scrollBy({top: d * window.innerHeight * f, behavior: 'smooth'});
})"""


def _scroll(page: Page, direction: int) -> str:
    before = page.evaluate("window.scrollY")
    page.evaluate(_SMOOTH_SCROLL, [direction, SCROLL_FRACTION, SCROLL_MAX_MS])
    after = page.evaluate("window.scrollY")
    if round(after) == round(before):
        return "already at the top" if direction < 0 else "already at the bottom"
    return f"scrolled from {round(before)}px to {round(after)}px"


def scroll_up(page: Page) -> str:
    return _scroll(page, -1)


def scroll_down(page: Page) -> str:
    return _scroll(page, +1)


def back(page: Page) -> str:
    url = page.url
    # Wait only until the navigation commits: pages restored from Chrome's
    # back/forward cache never fire a fresh "loaded" event.
    try:
        page.go_back(wait_until="commit", timeout=NAV_TIMEOUT_MS)
    except PlaywrightTimeoutError:
        pass
    if page.url == url:
        return "no previous page"
    return f"went back to {page.url}"


def tab(page: Page) -> str:
    page.evaluate(_ADD_HIGHLIGHT, HIGHLIGHT_CSS)
    if not page.evaluate(_FOCUS_NEXT):
        return "nothing to highlight"
    focused = page.evaluate(_DESCRIBE_FOCUSED)
    return f"highlighted {focused}" if focused else "nothing to highlight"


def click(page: Page) -> str:
    focused = page.evaluate(_DESCRIBE_FOCUSED)
    if focused is None:
        return "nothing is highlighted, use tab first"
    element = page.evaluate_handle(_TARGET).as_element()
    url = page.url
    element.click(timeout=CLICK_TIMEOUT_MS)
    page.wait_for_load_state("domcontentloaded")
    if page.url != url:
        return f"clicked {focused}, now on {page.url}"
    return f"clicked {focused}"


GESTURES = {
    "scroll_up": scroll_up,
    "scroll_down": scroll_down,
    "back": back,
    "tab": tab,
    "click": click,
}


def apply_gesture(page: Page, gesture: str) -> str:
    """Run the Playwright function for one gesture name."""
    try:
        action = GESTURES[gesture]
    except KeyError:
        raise ValueError(f"unknown gesture {gesture!r}, expected one of {list(GESTURES)}") from None
    return action(page)
