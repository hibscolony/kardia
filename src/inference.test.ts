import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { validateDataset, softmax, ensemble, scaleRR } from "./inference";
const demo = JSON.parse(readFileSync("public/data/demo.json", "utf8"));
test("accepts original notebook tensors and rejects raw or malformed signals", () => {
  assert.equal(validateDataset(demo).ecg.length, demo.ecg.length);
  for (const edit of [
    { rrFormat: "scaled" },
    { samplingRate: 250 },
    { ecg: [[[NaN]]] },
    { rr: [] },
    { version: "raw-ecg" },
    { annotations: ["bogus"] },
  ])
    assert.throws(() => validateDataset({ ...demo, ...edit }));
  const d = structuredClone(demo);
  d.ecg[0][0][0] = Infinity;
  assert.throws(() => validateDataset(d));
});
test("stable softmax and probability ensemble apply bias after averaging", () => {
  assert.deepEqual(softmax([1000, 1000]), [0.5, 0.5]);
  const p = ensemble(
    [
      [Math.log(0.8), Math.log(0.2)],
      [Math.log(0.4), Math.log(0.6)],
    ],
    [0, Math.log(2)],
  );
  assert.ok(Math.abs(p[0] - 3 / 7) < 1e-12);
  assert.ok(Math.abs(p[1] - 4 / 7) < 1e-12);
});
test("RR normalization follows float32 StandardScaler transform and clipping", () => {
  const s = { mean: [1, 2, 3], scale: [2, 1, 0.5] };
  assert.deepEqual(Array.from(scaleRR([3, 100, -100], s)), [1, 8, -8]);
});
