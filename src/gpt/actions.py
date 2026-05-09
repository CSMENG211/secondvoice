import os
from pathlib import Path
from time import sleep

from loguru import logger

from automation import activate_chrome, connect_to_cdp_browser
from automation.constants import DEFAULT_CDP_URL
from gpt.constants import (
    CHATGPT_COLOR_SCHEME,
    CHATGPT_RESPONSE_WAIT_TIMEOUT_MS,
    CHATGPT_SHORT_SCROLL_COUNT,
    CHATGPT_SHORT_SCROLL_DELTA_Y,
    CHATGPT_SHORT_SCROLL_PAUSE_SECONDS,
    CHATGPT_URL,
    SECONDVOICE_BADGE_ID,
    SECONDVOICE_CHATGPT_TAB_NAME,
    SECONDVOICE_TITLE_PREFIX,
)


def submit_to_chatgpt(
    prompt: str,
    photo_path: Path | None = None,
    cdp_url: str = DEFAULT_CDP_URL,
) -> tuple[bool, str | None]:
    """Open ChatGPT, place the prompt into the composer, and submit it."""
    if not prompt.strip():
        logger.info("Skipping ChatGPT submission because the transcript is empty.")
        return False, None

    suppress_node_deprecation_warnings()

    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        session = connect_to_cdp_browser(playwright, cdp_url)

        try:
            page = open_chatgpt_page(session.context)
            stabilize_chatgpt_theme(page)

            try:
                prompt_box = find_prompt_box(page, timeout=5_000)
            except PlaywrightTimeoutError:
                logger.warning("Could not find the ChatGPT prompt box yet.")
                logger.warning("If ChatGPT is asking you to log in, finish logging in inside the browser.")
                input("Press ENTER here after ChatGPT is open and ready for a prompt.")
                page.goto(CHATGPT_URL, wait_until="domcontentloaded")
                prompt_box = find_prompt_box(page, timeout=60_000)

            has_photo = photo_path is not None
            if has_photo:
                if not attach_file(page, photo_path):
                    logger.error("Skipping ChatGPT submission because the photo could not be attached.")
                    return False, None

            fill_prompt(prompt_box, prompt)
            submit_prompt(page, prompt_box, wait_for_upload=has_photo)
            logger.info("Submitted transcript to ChatGPT.")
            stop_auto_scroll_to_bottom(page)
            force_scroll_to_bottom(page)
            wait_for_chatgpt_response(page)
            scroll_down_short_times(page)
            activate_chrome()
            return True, latest_assistant_message(page)
        except PlaywrightTimeoutError as exc:
            logger.error("Could not find the ChatGPT prompt box.")
            logger.error("If this is the first run, log in to ChatGPT in the opened browser, then run again.")
            raise SystemExit(1) from exc
        finally:
            session.close()


def latest_assistant_message(page) -> str | None:
    """Return the latest visible assistant message text when available."""
    selectors = [
        '[data-message-author-role="assistant"]',
        'article[data-testid^="conversation-turn-"]',
        "main article",
    ]
    for selector in selectors:
        try:
            text = page.evaluate(
                """
                (sel) => {
                  const nodes = Array.from(document.querySelectorAll(sel));
                  if (!nodes.length) return null;
                  const node = nodes[nodes.length - 1];
                  return (node.innerText || "").trim() || null;
                }
                """,
                selector,
            )
            if text:
                return text
        except Exception:
            continue
    return None


def suppress_node_deprecation_warnings() -> None:
    """Suppress noisy Node deprecation warnings emitted by Playwright's driver."""
    node_options = os.environ.get("NODE_OPTIONS", "")
    no_deprecation_flag = "--no-deprecation"
    if no_deprecation_flag in node_options.split():
        return

    os.environ["NODE_OPTIONS"] = (
        f"{node_options} {no_deprecation_flag}".strip()
        if node_options
        else no_deprecation_flag
    )


def open_chatgpt_page(context):
    """Reuse the dedicated SecondVoice ChatGPT tab or open one when needed."""
    for page in context.pages:
        if page.url.startswith(CHATGPT_URL) and is_secondvoice_chatgpt_page(page):
            logger.info("Reusing dedicated SecondVoice ChatGPT tab.")
            mark_secondvoice_chatgpt_page(page)
            page.bring_to_front()
            activate_chrome()
            return page

    page = context.new_page()
    page.goto(CHATGPT_URL, wait_until="domcontentloaded")
    mark_secondvoice_chatgpt_page(page)
    return page


def is_secondvoice_chatgpt_page(page) -> bool:
    """Return whether a ChatGPT tab belongs to SecondVoice automation."""
    try:
        return page.evaluate(
            """
            (name) => window.name === name || sessionStorage.getItem(name) === "1"
            """,
            SECONDVOICE_CHATGPT_TAB_NAME,
        )
    except Exception as exc:
        logger.debug("Could not inspect ChatGPT tab marker: {}", exc)
        return False


def mark_secondvoice_chatgpt_page(page) -> None:
    """Mark one ChatGPT tab as the dedicated SecondVoice automation tab."""
    try:
        page.evaluate(
            """
            ({ name, badgeId, titlePrefix }) => {
              window.name = name;
              sessionStorage.setItem(name, "1");

              const updateTitle = () => {
                if (!document.title.startsWith(`${titlePrefix} - `)) {
                  document.title = `${titlePrefix} - ${document.title}`;
                }
              };
              updateTitle();
              window.__secondvoiceTitleTimer ||= window.setInterval(updateTitle, 1000);

              let badge = document.getElementById(badgeId);
              if (!badge) {
                badge = document.createElement("div");
                badge.id = badgeId;
                badge.textContent = titlePrefix;
                Object.assign(badge.style, {
                  position: "fixed",
                  right: "12px",
                  bottom: "12px",
                  zIndex: "2147483647",
                  padding: "6px 8px",
                  border: "1px solid rgba(255, 255, 255, 0.22)",
                  borderRadius: "6px",
                  background: "rgba(16, 16, 16, 0.86)",
                  color: "white",
                  font: "12px system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
                  pointerEvents: "none",
                });
                document.body.appendChild(badge);
              }
            }
            """,
            {
                "name": SECONDVOICE_CHATGPT_TAB_NAME,
                "badgeId": SECONDVOICE_BADGE_ID,
                "titlePrefix": SECONDVOICE_TITLE_PREFIX,
            },
        )
    except Exception as exc:
        logger.debug("Could not mark SecondVoice ChatGPT tab: {}", exc)


def stabilize_chatgpt_theme(page) -> None:
    """Keep ChatGPT's visual theme stable during automation."""
    page.emulate_media(color_scheme=CHATGPT_COLOR_SCHEME)
    try:
        page.evaluate(
            """
            (theme) => {
              for (const key of Object.keys(localStorage)) {
                if (key.startsWith("oai/apps/chatTheme/")) {
                  localStorage.setItem(key, JSON.stringify(theme));
                }
              }
              document.documentElement.classList.remove("light", "dark");
              document.documentElement.classList.add(theme);
              document.documentElement.style.colorScheme = theme;
            }
            """,
            CHATGPT_COLOR_SCHEME,
        )
    except Exception as exc:
        logger.debug("Could not stabilize ChatGPT theme: {}", exc)


def stop_auto_scroll_to_bottom(page) -> None:
    """Stop any old SecondVoice auto-scroll loop left on the ChatGPT page."""
    try:
        page.evaluate(
            """
            () => {
              const state = window.__secondvoiceAutoScroll;
              if (state?.timer) {
                window.clearInterval(state.timer);
              }
              if (state?.timeout) {
                window.clearTimeout(state.timeout);
              }
              if (state?.observer) {
                state.observer.disconnect();
              }
              delete window.__secondvoiceAutoScroll;
            }
            """
        )
    except Exception as exc:
        logger.debug("Could not stop ChatGPT auto-scroll: {}", exc)


def force_scroll_to_bottom(page) -> None:
    """Aggressively move every likely ChatGPT scroll container to the bottom."""
    try:
        page.evaluate(
            """
            () => {
              const containers = [
                document.scrollingElement,
                document.documentElement,
                document.body,
                document.querySelector("main"),
                document.querySelector("[role='main']"),
                ...document.querySelectorAll("div"),
              ].filter((element) => {
                if (!element) {
                  return false;
                }
                return element.scrollHeight > element.clientHeight + 8;
              });

              for (const element of containers) {
                element.scrollTop = element.scrollHeight;
              }
              window.scrollTo(0, document.body.scrollHeight);
            }
            """
        )
    except Exception as exc:
        logger.debug("Could not force-scroll ChatGPT to the bottom: {}", exc)


def wait_for_chatgpt_response(page) -> None:
    """Wait until ChatGPT appears to finish responding, with a timeout fallback."""
    try:
        stop_button = find_stop_button(page)
        if stop_button is not None:
            stop_button.wait_for(state="hidden", timeout=CHATGPT_RESPONSE_WAIT_TIMEOUT_MS)
            return

        page.wait_for_function(
            """
            () => {
              const button = document.querySelector(
                '[data-testid="send-button"], '
                + 'button[aria-label="Send prompt"], '
                + 'button[aria-label*="Send"]'
              );
              return button && !button.disabled && button.getAttribute("aria-disabled") !== "true";
            }
            """,
            timeout=CHATGPT_RESPONSE_WAIT_TIMEOUT_MS,
        )
    except Exception as exc:
        logger.debug("Timed out or could not detect ChatGPT response completion: {}", exc)


def find_stop_button(page):
    """Return ChatGPT's stop button while a response is streaming, if visible."""
    selectors = [
        "[data-testid='stop-button']",
        "button[aria-label='Stop generating']",
        "button[aria-label*='Stop']",
    ]

    for selector in selectors:
        button = page.locator(selector).first
        try:
            button.wait_for(state="visible", timeout=2_000)
            return button
        except Exception:
            continue

    return None


def scroll_down_short_times(page) -> None:
    """Nudge the ChatGPT page downward a few times without pinning it."""
    try:
        for _ in range(CHATGPT_SHORT_SCROLL_COUNT):
            page.evaluate(
                """
                (deltaY) => {
                  const containers = [
                    document.scrollingElement,
                    document.documentElement,
                    document.body,
                    document.querySelector("main"),
                    document.querySelector("[role='main']"),
                    ...document.querySelectorAll("div"),
                  ].filter((element) => {
                    if (!element) {
                      return false;
                    }
                    const style = window.getComputedStyle(element);
                    const canScroll = /(auto|scroll)/.test(style.overflowY);
                    return canScroll && element.scrollHeight > element.clientHeight + 8;
                  });

                  for (const element of containers) {
                    element.scrollBy({ top: deltaY, behavior: "smooth" });
                  }
                  window.scrollBy({ top: deltaY, behavior: "smooth" });
                }
                """,
                CHATGPT_SHORT_SCROLL_DELTA_Y,
            )
            sleep(CHATGPT_SHORT_SCROLL_PAUSE_SECONDS)
    except Exception as exc:
        logger.debug("Could not short-scroll ChatGPT: {}", exc)


def fill_prompt(prompt_box, prompt: str) -> None:
    """Fill the prompt box without submitting the message."""
    prompt_box.click()
    prompt_box.fill(prompt)


def submit_prompt(page, prompt_box, wait_for_upload: bool = False) -> None:
    """Submit the composed message after ChatGPT enables sending."""
    if wait_for_upload:
        wait_for_attachment_upload(page)

    send_button = find_send_button(page)
    if send_button is None:
        logger.warning("Could not find ChatGPT send button; falling back to Enter.")
        prompt_box.press("Enter")
        return

    send_button.click(timeout=30_000)


def find_send_button(page):
    """Return ChatGPT's send button when one of the known selectors is visible."""
    selectors = [
        "[data-testid='send-button']",
        "button[aria-label='Send prompt']",
        "button[aria-label*='Send']",
    ]

    for selector in selectors:
        button = page.locator(selector).first
        try:
            button.wait_for(state="visible", timeout=2_000)
            return button
        except Exception:
            continue

    return None


def attach_file(page, file_path: Path) -> bool:
    """Attach a local file to the current ChatGPT composer."""
    if not file_path.exists() or not file_path.is_file():
        logger.warning("Skipping ChatGPT photo upload because the file is missing: {}", file_path)
        return False

    logger.info("Uploading interview photo to ChatGPT: {}", file_path)

    file_inputs = page.locator("input[type='file']")
    if file_inputs.count() > 0:
        file_inputs.last.set_input_files(str(file_path))
        wait_for_attachment_upload(page)
        return True

    attach_selectors = [
        "[data-testid='composer-plus-btn']",
        "button[aria-label*='Attach']",
        "button[aria-label*='Upload']",
        "button:has-text('Attach')",
        "button:has-text('Upload')",
    ]

    for selector in attach_selectors:
        button = page.locator(selector).first
        try:
            button.wait_for(state="visible", timeout=2_000)
            with page.expect_file_chooser(timeout=5_000) as chooser_info:
                button.click()
            chooser_info.value.set_files(str(file_path))
            wait_for_attachment_upload(page)
            return True
        except Exception:
            continue

    logger.warning("Could not find a ChatGPT file upload control.")
    return False


def wait_for_attachment_upload(page, timeout: int = 30_000) -> None:
    """Wait briefly for ChatGPT to finish accepting the selected file."""
    try:
        page.get_by_text("Uploading", exact=False).first.wait_for(state="hidden", timeout=timeout)
    except Exception:
        # Some ChatGPT builds do not expose upload text; the send button click
        # still auto-waits for the composer to become actionable.
        sleep(1)


def find_prompt_box(page, timeout: int):
    """Find the current ChatGPT message composer using known selectors."""
    selectors = [
        "[data-testid='prompt-textarea']",
        "#prompt-textarea",
        "textarea[placeholder*='Message']",
        "div[contenteditable='true']",
    ]

    for selector in selectors:
        prompt_box = page.locator(selector).first
        try:
            prompt_box.wait_for(state="visible", timeout=timeout)
            return prompt_box
        except Exception:
            continue

    page.wait_for_selector(",".join(selectors), state="visible", timeout=timeout)
    return page.locator(",".join(selectors)).first
