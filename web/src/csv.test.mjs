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

const ticks = (header) => numericColumns(table(header, (i) => String(i)))
  .map((c) => [c.name, c.include]);

test("row numbers, account numbers and contact columns start unticked", () => {
  const header = ["#", "No", "No.", "S/N", "Sno", "SrNo", "RowNumber", "Unnamed: 0.1", "index", "idx", "row_index",
    "IDNumber", "id_number", "customer_no", "account_number", "AccountNo", "roll_no", "emp_no",
    "order_number", "ticket_no", "UID", "uid", "user_uid", "mobile", "mobile_no", "telephone",
    "contact_number", "pin", "PIN"];
  assert.deepStrictEqual(ticks(header), header.map((name) => [name, false]));
});

test("a name that starts with id or uid is an id", () => {
  const header = ["id_user", "ID_Customer", "id-number", "Id User", "uid_device", "UID-Account"];
  assert.deepStrictEqual(ticks(header), header.map((name) => [name, false]));
});

test("a name that starts with id but ends in a measure is a measure", () => {
  const header = ["id_count", "ID Count", "uid_count", "ID Card Count", "ID Score", "id_score", "ID_Weight",
    "id_total", "IdAvg", "id_number_of_visits"];
  assert.deepStrictEqual(ticks(header), header.map((name) => [name, true]));
});

test("a week, month or version number is a measure, not a row label", () => {
  const header = ["Week Number", "Month Number", "week_num", "day_number", "Version Number", "WeekNo",
    "year_no", "Week #", "# of visits"];
  assert.deepStrictEqual(ticks(header), header.map((name) => [name, true]));
});

test("a bare row or sequence column, and a number written with #, is an id", () => {
  const header = ["Row", "row", "seq", "Seq", "SeqNo", "Cust#", "Item #", "Order#"];
  assert.deepStrictEqual(ticks(header), header.map((name) => [name, false]));
});

test("measures whose names contain an id-like word stay ticked", () => {
  const header = ["paid", "valid", "video", "grid", "solid", "android", "width", "bid", "kid_count",
    "isPaid", "amountPaid", "Lipid", "avgHumidity", "phone_minutes", "amount", "kids",
    "number_of_visits", "num_items", "pin_count", "price_index", "uv_index", "UVIndex", "gini_index",
    "happiness_index", "BodyMassIndex", "Index", "idle_time", "ideal_weight", "identity_score", "IdleTime",
    "idx_score", "uids_seen", "fluid_intake", "Idea Count"];
  assert.deepStrictEqual(ticks(header), header.map((name) => [name, true]));
});

test("uploads above the size cap are refused before they are read", () => {
  assert.strictEqual(uploadSizeError(MAX_UPLOAD_BYTES), null);
  assert.match(uploadSizeError(100 * 1024 * 1024), /^That file is 100 MB\..* capped at 20 MB /);
});

test("a file just over the cap reads as over it, not as the cap itself", () => {
  const kb = 1024;
  for (const [bytes, shown] of [[MAX_UPLOAD_BYTES + 1, "20.1"], [MAX_UPLOAD_BYTES + 400 * kb, "20.4"],
    [MAX_UPLOAD_BYTES + 460 * kb, "20.5"], [MAX_UPLOAD_BYTES + 1024 * kb - 1, "21"]]) {
    assert.ok(uploadSizeError(bytes).startsWith(`That file is ${shown} MB.`), `${bytes}: ${uploadSizeError(bytes)}`);
  }
});

test("NA, NaN, null, #N/A, None, n.a., ?, - and . are missing values like a blank, not text", () => {
  for (const marker of ["NA", "N/A", "NaN", "nan", "null", "NULL", "?", "-", " na ", "#N/A", "None", "n.a.",
    "N.A.", "."]) {
    const rows = table(["a", "b"], (i, c) => (c === 1 && i === 5 ? marker : String(i - c * 10)));
    const cols = numericColumns(rows);
    assert.deepStrictEqual(cols.map((c) => c.name), ["a", "b"], marker);
    const m = matrix(rows, cols);
    assert.strictEqual(m.length, 59, marker);
    assert.ok(!m.some(([a]) => a === 5), marker);
  }
});

test("a lone - is missing but -5 is a number", () => {
  assert.strictEqual(toNumber("-5"), -5);
  assert.ok(Number.isNaN(toNumber("-")));
  const rows = [["x", "y"], ...Array.from({ length: 10 }, (_, i) => [i === 2 ? "-" : `-${i + 1}`, String(i)])];
  const cols = numericColumns(rows);
  assert.deepStrictEqual(cols.map((c) => c.name), ["x", "y"]);
  assert.deepStrictEqual(matrix(rows, cols).map(([x]) => x), [-1, -2, -4, -5, -6, -7, -8, -9, -10]);
});

test("a lone . is missing but .5 is a number", () => {
  assert.strictEqual(toNumber(".5"), 0.5);
  assert.ok(Number.isNaN(toNumber(".")));
  const rows = [["x", "y"], ...Array.from({ length: 10 }, (_, i) => [i === 2 ? "." : `.${i + 1}`, String(i)])];
  const cols = numericColumns(rows);
  assert.deepStrictEqual(cols.map((c) => c.name), ["x", "y"]);
  assert.deepStrictEqual(matrix(rows, cols).map(([x]) => x), [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1]);
});

test("text that is not a missing marker still hides a column", () => {
  const rows = table(["a", "b"], (i, c) => (c === 1 && i === 5 ? "none of it" : String(i)));
  assert.deepStrictEqual(numericColumns(rows).map((c) => c.name), ["a"]);
});
