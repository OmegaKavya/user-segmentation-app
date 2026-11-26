import test from "node:test";
import assert from "node:assert";
import { parseCsv, toNumber, numericColumns, matrix, uploadSizeError, MAX_UPLOAD_BYTES } from "./csv.js";

test("quoted fields keep their commas, quotes and newlines", () => {
  const rows = parseCsv('a,b,c\n"1,200",2,"say ""hi""\nthere"\r\n3,4,5\n');
  assert.deepStrictEqual(rows, [["a", "b", "c"], ["1,200", "2", 'say "hi"\nthere'], ["3", "4", "5"]]);
});

test("a byte order mark and blank lines are ignored", () => {
  assert.deepStrictEqual(parseCsv("﻿x,y\n\n1,2\n"), [["x", "y"], ["1", "2"]]);
});

test("thousands separators are numbers; text is not", () => {
  assert.strictEqual(toNumber("1,200"), 1200);
  assert.strictEqual(toNumber(" 3.5 "), 3.5);
  assert.ok(Number.isNaN(toNumber("abc")));
  assert.ok(Number.isNaN(toNumber("")));
});

test("identifier-like columns are found but left out by default", () => {
  const rows = parseCsv("user_id,age,income,zip,name\n1,30,\"1,000\",500001,a\n2,40,2000,500002,b\n");
  const cols = numericColumns(rows);
  assert.deepStrictEqual(cols.map((c) => [c.name, c.include]),
    [["user_id", false], ["age", true], ["income", true], ["zip", false]]);
  assert.deepStrictEqual(matrix(rows, cols.filter((c) => c.include)), [[30, 1000], [40, 2000]]);
});

test("an unclosed quote is an error that names its line, not one giant cell", () => {
  assert.throws(() => parseCsv('a,b\n1,2\n"3,4\n5,6\n'), /quote opened on line 3 is never closed/);
});

test("only finite decimal numbers count, not hex, binary, octal or Infinity", () => {
  for (const cell of ["0x1A", "0b101", "0o17", "Infinity", "-Infinity", "1e999", "1_000"]) {
    assert.ok(Number.isNaN(toNumber(cell)), cell);
  }
  assert.strictEqual(toNumber("1e3"), 1000);
  assert.strictEqual(toNumber("-2.5E-2"), -0.025);
  assert.strictEqual(toNumber("+.5"), 0.5);
  assert.strictEqual(toNumber("5."), 5);
  const rows = [["h", "x"], ...Array.from({ length: 10 }, (_, i) => [`0x${i + 10}`, String(i)])];
  assert.deepStrictEqual(numericColumns(rows).map((c) => c.name), ["x"]);
});

const table = (header, cell, n = 60) =>
  [header, ...Array.from({ length: n }, (_, i) => header.map((_, c) => cell(i, c)))];

test("a blank cell is a missing value, not text, and its row is left out", () => {
  const rows = table(["a", "b", "c"], (i, c) => (c === 1 && i === 3 ? " " : String((i * (c + 2)) % 7)));
  assert.deepStrictEqual(numericColumns(rows).map((c) => c.name), ["a", "b", "c"]);
  assert.strictEqual(matrix(rows, numericColumns(rows)).length, 59);
});

test("a column that is mostly blank is not offered", () => {
  const rows = table(["a", "sparse"], (i, c) => (c === 1 && i % 3 !== 0 ? "" : String(i)));
  assert.deepStrictEqual(numericColumns(rows).map((c) => c.name), ["a"]);
});

test("row numbers written by spreadsheets and pandas start unticked", () => {
  const header = ["", "Unnamed: 0", "userId", "CustomerID", "S.No", "Sr. No.", "serial_number", "zip_code",
    "paid", "video", "valid", "kids", "amount"];
  const cols = numericColumns(table(header, (i) => String(i)));
  assert.deepStrictEqual(cols.map((c) => [c.name, c.include]), [
    ["column 1", false], ["Unnamed: 0", false], ["userId", false], ["CustomerID", false], ["S.No", false],
    ["Sr. No.", false], ["serial_number", false], ["zip_code", false],
    ["paid", true], ["video", true], ["valid", true], ["kids", true], ["amount", true],
  ]);
});

test("uploads above the size cap are refused before they are read", () => {
  assert.strictEqual(uploadSizeError(MAX_UPLOAD_BYTES), null);
  assert.match(uploadSizeError(MAX_UPLOAD_BYTES + 1), /20 MB/);
  assert.match(uploadSizeError(100 * 1024 * 1024), /100 MB/);
});
