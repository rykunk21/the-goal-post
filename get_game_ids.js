#!/usr/bin/env node
/**
 * get_game_ids.js
 * ---------------
 * Uses Puppeteer to load a StatBroadcast archive page, waits for the
 * #archiveTable to populate via AJAX, then extracts game IDs from the
 * "archived.php?id=XXXXXX" links in each row.
 *
 * Usage:
 *   node get_game_ids.js <gid> [n_games]
 *
 * Output:
 *   JSON array of game ID strings to stdout, e.g. ["612260","611843",...]
 *   Errors/logs go to stderr so they don't pollute the JSON output.
 *
 * Install:
 *   npm install puppeteer
 */

const puppeteer = require("puppeteer");

const gid     = process.argv[2];
const nGames  = parseInt(process.argv[3] || "5", 10);

if (!gid) {
  console.error("Usage: node get_game_ids.js <gid> [n_games]");
  process.exit(1);
}

const PAGE_URL = `https://www.statbroadcast.com/events/archive.php?gid=${gid}`;

(async () => {
  console.error(`[${gid}] Launching Puppeteer → ${PAGE_URL}`);

  const browser = await puppeteer.launch({
    headless: "new",
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
  });

  try {
    const page = await browser.newPage();

    await page.setUserAgent(
      "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 " +
      "(KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
    );

    // We need to intercept the _archive.php AJAX call that the page fires
    // after the sport filter is selected. That call includes a computed
    // time= and hash= in its URL which authenticates it — we cannot
    // construct these ourselves, we must capture the live request.
    //
    // Strategy:
    //   1. Intercept all responses from _archive.php
    //   2. Trigger the Men's Basketball filter to fire the AJAX call
    //   3. Capture the JSON response body directly — it contains `eventid`
    //      in every row, no clicking required.

    let capturedData = null;

    page.on("response", async (response) => {
      if (response.url().includes("_archive.php") && capturedData === null) {
        try {
          const json = await response.json();
          if (json && json.data && json.data.length > 0) {
            capturedData = json;
            console.error(`[${gid}] Intercepted AJAX response — ${json.data.length} rows`);
          }
        } catch (e) {
          // not JSON or already consumed
        }
      }
    });

    // Navigate — the initial page load fires a first AJAX call (all sports)
    await page.goto(PAGE_URL, { waitUntil: "networkidle2", timeout: 30000 });

    // If initial load already gave us basketball data, great.
    // Otherwise select the sport filter to trigger a fresh AJAX call.
    if (!capturedData || capturedData.data[0].sport !== "Men's Basketball") {
      capturedData = null; // reset so we capture the filtered response
      console.error(`[${gid}] Selecting Men's Basketball filter…`);
      await page.select("#sports", "M;bbgame");
      await page.waitForNetworkIdle({ idleTime: 1500, timeout: 15000 }).catch(() => {
        console.error(`[${gid}] WARNING: network did not go idle after filter`);
      });
    }

    if (!capturedData) {
      throw new Error(`No AJAX data captured for '${gid}' — the filter may not have fired`);
    }

    // Extract eventid from each row — it's a top-level field in the JSON
    const gameIds = capturedData.data
      .slice(0, nGames)
      .map(row => String(row.eventid))
      .filter(id => id && id !== "undefined");

    console.error(`[${gid}] Found ${gameIds.length} game IDs: ${gameIds.join(", ")}`);

    // Output clean JSON to stdout for the Python caller
    console.log(JSON.stringify(gameIds));

  } finally {
    await browser.close();
  }
})().catch((err) => {
  console.error(`[${gid}] Fatal error: ${err.message}`);
  process.exit(1);
});
