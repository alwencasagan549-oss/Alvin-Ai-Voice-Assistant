// Container-aware scrolling helpers.
//
// The dashboard has two different scroll models:
//
//   * Desktop (> 700px): `.app-shell` is a grid. `.main-content` is the scroll
//     container and the topbar lives in its own grid row, so nothing ever
//     overlaps the content column.
//   * Mobile (<= 700px): `.app-shell` becomes a flex column and the document
//     scrolls. The topbar and the nav strip are `position: sticky`, so they
//     cover the top of the viewport.
//
// `Element.scrollIntoView()` does not behave consistently across that split:
// it can silently leave the content column untouched on desktop and it does
// not move the window at all on mobile. Navigation therefore goes through
// these helpers, which compute the target offset explicitly.

const STICKY_SELECTORS = ['.topbar', '.sidebar'];

/**
 * Nearest ancestor that actually scrolls vertically, or null when the
 * document itself is the scroller.
 */
export function findScrollContainer(el) {
  let node = el?.parentElement;
  while (node && node !== document.body) {
    const overflowY = getComputedStyle(node).overflowY;
    if (overflowY === 'auto' || overflowY === 'scroll') return node;
    node = node.parentElement;
  }
  return null;
}

/**
 * Height of the sticky chrome that floats above the scrolled content.
 * Zero on desktop (the topbar is a separate grid row), the combined
 * topbar + nav height on mobile.
 */
export function stickyChromeHeight() {
  let total = 0;
  for (const selector of STICKY_SELECTORS) {
    const node = document.querySelector(selector);
    if (!node) continue;
    if (getComputedStyle(node).position !== 'sticky') continue;
    const rect = node.getBoundingClientRect();
    if (rect.height <= 0 || rect.bottom <= 0) continue;
    total += rect.height;
  }
  return total;
}

/**
 * Scroll `el` into view inside whichever container owns it.
 *
 * @param {Element} el            target element
 * @param {object}  options
 * @param {number}  options.offset  pixels of sticky chrome to clear
 * @param {number}  options.gap     extra breathing room below the chrome
 * @param {string}  options.behavior scroll behaviour
 */
export function scrollToElement(el, { offset = 0, gap = 0, behavior = 'smooth' } = {}) {
  if (!el) return;
  const container = findScrollContainer(el);
  const base = container || document.scrollingElement;
  if (!base) return;

  const baseRect = base.getBoundingClientRect();
  const paddingTop = container ? (parseFloat(getComputedStyle(base).paddingTop) || 0) : 0;
  const contentOffset = el.getBoundingClientRect().top - baseRect.top + base.scrollTop;
  const target = Math.max(0, contentOffset - paddingTop - gap - offset);

  if (container) container.scrollTo({ top: target, behavior });
  else window.scrollTo({ top: target, behavior });
}
