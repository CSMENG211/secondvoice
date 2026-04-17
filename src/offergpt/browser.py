from pathlib import Path
from time import sleep


CHATGPT_URL = "https://chatgpt.com/"
DEFAULT_BROWSER_PROFILE = Path.home() / ".offergpt" / "browser-profile"
TYPE_DELAY_MS = 25


def submit_to_chatgpt(prompt: str, profile_dir: Path = DEFAULT_BROWSER_PROFILE) -> None:
    if not prompt.strip():
        print("Skipping ChatGPT submission because the transcript is empty.")
        return

    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright

    profile_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = launch_browser(playwright, profile_dir)

        try:
            page = browser.new_page()
            page.goto(CHATGPT_URL, wait_until="domcontentloaded")
            page.wait_for_load_state("networkidle", timeout=30_000)

            try:
                prompt_box = find_prompt_box(page, timeout=15_000)
            except PlaywrightTimeoutError:
                print("Could not find the ChatGPT prompt box yet.")
                print("If ChatGPT is asking you to log in, finish logging in inside the browser.")
                input("Press ENTER here after ChatGPT is open and ready for a prompt.")
                page.goto(CHATGPT_URL, wait_until="domcontentloaded")
                prompt_box = find_prompt_box(page, timeout=60_000)

            prompt_box.click()
            sleep(0.5)
            prompt_box.press_sequentially(prompt, delay=TYPE_DELAY_MS)
            sleep(0.5)
            prompt_box.press("Enter")
            print("Submitted transcript to ChatGPT.")
            input("Browser is open. Press ENTER here when you are ready to close it.")
        except PlaywrightTimeoutError as exc:
            print("Could not find the ChatGPT prompt box.")
            print("If this is the first run, log in to ChatGPT in the opened browser, then run again.")
            raise SystemExit(1) from exc
        finally:
            browser.close()


def launch_browser(playwright, profile_dir: Path):
    launch_options = {
        "user_data_dir": str(profile_dir),
        "headless": False,
        "no_viewport": True,
        "args": [
            "--start-maximized",
            "--disable-blink-features=AutomationControlled",
        ],
    }

    try:
        print("Opening ChatGPT with installed Google Chrome...")
        return playwright.chromium.launch_persistent_context(
            channel="chrome",
            **launch_options,
        )
    except Exception as exc:
        print(f"Could not launch installed Chrome: {exc}")
        print("Falling back to Playwright Chromium...")
        return playwright.chromium.launch_persistent_context(**launch_options)


def find_prompt_box(page, timeout: int):
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
