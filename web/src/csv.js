/**
 * CSV parsing and column preparation for uploaded datasets.
 *
 * The upload used to split every line on every comma, so a quoted number
 * such as "1,200" became two cells and every column after it shifted by one.
 * This follows RFC 4180: fields may be quoted, quoted fields may contain
 * commas, newlines and doubled quotes ("").
 */

// Parsing holds every cell as a string at once, which measured about 13 times
// the file's size in heap (Node, a file of short numeric cells). At 20 MB that
// is under 300 MB; at 100 MB it is well over a gigabyte, which is enough for
// the browser to kill the tab, and sooner on a phone.
export const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;

// Rounded up to a tenth so a file just over the cap never reads as the cap
// itself ("20 MB ... capped at 20 MB"). Multiplying before dividing by a
// power of two keeps whole byte counts exact.
const megabytes = (bytes) => Math.ceil((bytes * 10) / (1024 * 1024)) / 10;

/** Why a file of this size is refused, or null when it can be read. */
export function uploadSizeError(bytes) {
  if (bytes <= MAX_UPLOAD_BYTES) return null;
  return `That file is ${megabytes(bytes)} MB. Uploads are parsed in your browser and are capped at `
    + `${megabytes(MAX_UPLOAD_BYTES)} MB so the tab does not run out of memory. A random sample of the `
    + "rows answers the same question.";
}

export function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  let line = 1;
  let quoteLine = 0;
  const src = text.replace(/^﻿/, "");

  for (let i = 0; i < src.length; i += 1) {
    const ch = src[i];
    if (quoted) {
      if (ch === '"') {
        if (src[i + 1] === '"') { field += '"'; i += 1; }
        else quoted = false;
      } else {
        if (ch === "\n" || (ch === "\r" && src[i + 1] !== "\n")) line += 1;
        field += ch;
      }
    } else if (ch === '"' && field === "") {
      quoted = true;
      quoteLine = line;
    } else if (ch === ",") {
      row.push(field); field = "";
    } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && src[i + 1] === "\n") i += 1;
      line += 1;
      row.push(field); field = "";
      rows.push(row); row = [];
    } else {
      field += ch;
    }
  }
  // Otherwise everything after a stray quote becomes one cell and the file
  // looks like it has a single short row, with no hint why.
  if (quoted) {
    throw new Error(`A quote opened on line ${quoteLine} is never closed, so the rest of the file `
      + "cannot be split into cells. Check that line for a stray \" character.");
  }
  if (field !== "" || row.length) { row.push(field); rows.push(row); }
  return rows.filter((r) => r.some((cell) => cell.trim() !== ""));
}

// Number() also accepts 0x1A, 0b101, 0o17 and Infinity, none of which is a
// measurement a CSV column means.
const DECIMAL = /^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/i;

/** A cell as a finite decimal number, allowing thousands separators, or NaN. */
export function toNumber(cell) {
  if (cell === undefined) return NaN;
  const plain = String(cell).trim().replace(/(\d),(?=\d{3}\b)/g, "$1");
  if (!DECIMAL.test(plain)) return NaN;
  const value = Number(plain);
  return Number.isFinite(value) ? value : NaN;
}

// What R (NA), pandas (NaN), databases (null) and hand-kept sheets ("?", "-")
// write for an empty cell. Read as text, one of them in the sample hid the
// whole column.
const MISSING_MARKERS = new Set(["", "na", "n/a", "nan", "null", "?", "-"]);

const isMissing = (cell) => cell === undefined || MISSING_MARKERS.has(String(cell).trim().toLowerCase());

// Columns that are numeric but are labels, not measurements. Clustering on a
// row id or a postcode finds structure in how the file was numbered. Names
// are split into words (snake_case, camelCase, "IDNumber", "S.No") so that
// "userId" is an id but "paid", "valid" and "Lipid" are not. Only the last
// word names what a column holds: "pin" is a PIN code, "pin_count" a measure.
// The one exception is a leading "id" or "uid", as in "id_user" or
// "ID_Customer", which names the column for the id before it says whose.
const IDENTIFIER_WORDS = new Set(["id", "uid", "uuid", "guid", "zip", "zipcode", "postcode",
  "postalcode", "pin", "pincode", "phone", "phonenumber", "mobile", "telephone"]);
// "customer_no", "account_number", "No." and "IDNumber" are numbers that
// label a row; a measure names what it counts last ("number_of_visits").
const NUMBER_WORDS = new Set(["no", "nr", "num", "number"]);
const LEADING_ID_WORDS = new Set(["id", "uid"]);
const SERIAL_NUMBER = /^((s|sr|sl|serial|row)(n|no|nr|num|number)|serial)$/;
// "price_index" and "BodyMassIndex" are measures, so index counts only as the
// whole name. pandas' reset_index() writes a lowercase "index"; a capitalised
// "Index" is as likely a score, so case matters for the bare word.
const isRowIndex = (raw, words) => raw === "index" || raw === "idx" || /^row (index|idx)$/.test(words.join(" "));

function looksLikeIdentifier(header) {
  const raw = header.trim();
  // A blank header or pandas' "Unnamed: 0" (or "Unnamed: 0.1" once such a
  // file is read and written again) is an index written by to_csv().
  if (raw === "" || /^unnamed: ?\d+(\.\d+)?$/i.test(raw)) return true;
  const words = raw.replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2").replace(/([a-z\d])([A-Z])/g, "$1 $2")
    .toLowerCase().split(/[^a-z\d]+/).filter(Boolean);
  // A header of punctuation only, such as "#", is a row number.
  if (words.length === 0) return true;
  const last = words.at(-1);
  return isRowIndex(raw, words) || IDENTIFIER_WORDS.has(last) || NUMBER_WORDS.has(last)
    || LEADING_ID_WORDS.has(words[0])
    || IDENTIFIER_WORDS.has(words.slice(-2).join(""))
    || SERIAL_NUMBER.test(words.join(""));
}

/**
 * The numeric columns of a parsed CSV, each marked with whether it should be
 * included by default. Blank cells and markers such as NA are missing values,
 * so a column counts as numeric when every filled cell in the sample is a
 * number, provided at
 * least half the sample is filled; a mostly empty column would cost most of
 * the rows if ticked.
 */
export function numericColumns(rows) {
  const [header, ...body] = rows;
  const sample = body.slice(0, 40);
  return header
    .map((name, index) => ({
      name: name.trim() || `column ${index + 1}`, index, include: !looksLikeIdentifier(name),
    }))
    .filter(({ index }) => {
      const filled = sample.map((r) => r[index]).filter((cell) => !isMissing(cell));
      return filled.length > 0 && filled.length * 2 >= sample.length
        && filled.every((cell) => !Number.isNaN(toNumber(cell)));
    });
}

/**
 * The rows complete in every chosen column, as numbers. A row with a missing
 * or non-numeric cell in any chosen column is left out rather than imputed.
 */
export function matrix(rows, columns) {
  return rows.slice(1)
    .map((r) => columns.map((c) => toNumber(r[c.index])))
    .filter((values) => values.every((v) => Number.isFinite(v)));
}
