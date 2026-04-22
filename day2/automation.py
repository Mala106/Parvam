"""
automation.py
Simple Selenium script to log in to the provided URL using XPath selectors.

Usage:
  python automation.py --email-value you@example.com --password-value secret

Options:
  --url            Login page URL (defaults to scholar.parvam.in login)
  --email-value    Email string to type into the email field
  --password-value Password string to type into the password field
  --headless       Run browser headless (no UI)

This script uses Selenium and webdriver-manager to launch Chrome.
Install dependencies: pip install -r requirements.txt
"""

import argparse
import sys
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

try:
    # webdriver-manager simplifies driver management
    from webdriver_manager.chrome import ChromeDriverManager
    HAS_WDM = True
except Exception:
    HAS_WDM = False


DEFAULT_URL = "https://scholar.parvam.in/student/login"

# XPaths provided by the user
EMAIL_XPATH = "/html/body/div[1]/div/div/div/div[2]/div/div/form/div/div[2]/div/div[2]/div[1]/input"
PASSWORD_XPATH = "/html/body/div[1]/div/div/div/div[2]/div/div/form/div/div[2]/div/div[2]/div[2]/input"
BUTTON_XPATH = "/html/body/div[1]/div/div/div/div[2]/div/div/form/div/div[2]/div/div[4]/button"


def build_driver(headless: bool = False):
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    if HAS_WDM:
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
    else:
        # Fall back to system chromedriver in PATH
        driver = webdriver.Chrome(options=options)
    return driver


def login(url: str, email: str, password: str, headless: bool = False, timeout: int = 15):
    driver = build_driver(headless=headless)
    wait = WebDriverWait(driver, timeout)
    try:
        driver.get(url)

        # Wait for email field and enter value
        email_el = wait.until(EC.presence_of_element_located((By.XPATH, EMAIL_XPATH)))
        email_el.clear()
        email_el.send_keys(email)

        # Wait for password field and enter value
        password_el = wait.until(EC.presence_of_element_located((By.XPATH, PASSWORD_XPATH)))
        password_el.clear()
        password_el.send_keys(password)

        # Wait for the button and click
        btn = wait.until(EC.element_to_be_clickable((By.XPATH, BUTTON_XPATH)))
        btn.click()

        # Give the page a moment to respond
        time.sleep(2)

        # Try to detect successful navigation by checking URL or title change
        current_url = driver.current_url
        page_title = driver.title
        print(f"After click: url={current_url}, title={page_title}")

    except TimeoutException as e:
        print("Timeout waiting for page elements:", e)
        driver.save_screenshot("login_timeout.png")
        raise
    finally:
        driver.quit()


def main():
    parser = argparse.ArgumentParser(description="Automate login using XPath selectors")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--email-value", required=True)
    parser.add_argument("--password-value", required=True)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    try:
        login(args.url, args.email_value, args.password_value, headless=args.headless)
    except Exception as exc:
        print("Login automation failed:", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
