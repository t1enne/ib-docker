import { delay } from "es-toolkit";
import { chromium } from "playwright";

async function loginToIBKR(username: string, password: string) {
  const browser = await chromium.launch({ headless: true }); // Set to true for headless
  const context = await browser.newContext({
    ignoreHTTPSErrors: true, // Handle self-signed cert
  });
  const page = await context.newPage();

  try {
    await page.goto("https://localhost:5000");

    // set paper trading mode
    await page.click('label[for="toggle1"]');
    await delay(500);
    await page.fill(
      'input[name="username"], #username, input[type="text"]',
      username,
    );
    await page.fill(
      'input[name="password"], #password, input[type="password"]',
      password,
    );
    await page.click(
      'button[type="submit"], input[type="submit"], .login-button',
    );

    // Wait for navigation or success indicator
    await page.waitForLoadState("networkidle");

    console.log("Login attempt completed");
  } catch (error) {
    console.error("Login failed:", error);
  } finally {
    // Optionally close browser, or keep it open for manual interaction
    await browser.close();
  }
}

// Get credentials from environment variables or command line args
const username = process.env.IBKR_USERNAME || process.argv[2];
const password = process.env.IBKR_PASSWORD || process.argv[3];

if (!username || !password) {
  console.error(
    "Please provide username and password via env vars IBKR_USERNAME/IBKR_PASSWORD or command line args",
  );
  process.exit(1);
}

loginToIBKR(username, password);
