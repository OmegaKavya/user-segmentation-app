import test from "node:test";
import assert from "node:assert";
import { parseCsv, toNumber, numericColumns, matrix } from "./csv.js";

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
