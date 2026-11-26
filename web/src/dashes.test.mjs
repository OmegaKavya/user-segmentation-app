import test from "node:test";
import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

// The page's copy uses no en or em dashes; a missing number once showed as a
// bare em dash, which reads as a glitch rather than as "no value".
const DASH = /[\u2013\u2014]|&(n|m)dash;/;
const web = path.dirname(path.dirname(fileURLToPath(import.meta.url)));

function sources(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) return sources(full);
    return /\.(jsx?|mjs|css|json|html)$/.test(entry.name) ? [full] : [];
  });
}

test("nothing the page renders contains an en or em dash", () => {
  const files = [...sources(path.join(web, "src")), path.join(web, "index.html")];
  const hits = files.flatMap((file) => fs.readFileSync(file, "utf8").split("\n")
    .flatMap((line, i) => (DASH.test(line) ? [`${path.relative(web, file)}:${i + 1}`] : [])));
  assert.deepStrictEqual(hits, []);
});
