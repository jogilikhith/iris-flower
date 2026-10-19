"use strict";
const $ = id => document.getElementById(id);
const COLORS = ["#d9930d", "#1f8a8a", "#6b46c1"];
const LABELS = ["Sepal length", "Sepal width", "Petal length", "Petal width"];

// Plot geometry: petal length on x, petal width on y.
const W = 480, H = 400, M = { l: 52, r: 16, t: 12, b: 48 };
const XR = [0.5, 7.2], YR = [0, 2.8];
const sx = v => M.l + ((v - XR[0]) / (XR[1] - XR[0])) * (W - M.l - M.r);
const sy = v => H - M.b - ((v - YR[0]) / (YR[1] - YR[0])) * (H - M.t - M.b);

let model, values;

function drawPlot() {
  let s = "";
  for (let x = 1; x <= 7; x++) {
    s += `<line class="grid" x1="${sx(x)}" x2="${sx(x)}" y1="${sy(YR[0])}" y2="${sy(YR[1])}"/>
          <text class="tick" x="${sx(x)}" y="${H - M.b + 18}" text-anchor="middle">${x}</text>`;
  }
  for (let y = 0; y <= 2.5; y += 0.5) {
    s += `<line class="grid" x1="${M.l}" x2="${W - M.r}" y1="${sy(y)}" y2="${sy(y)}"/>
          <text class="tick" x="${M.l - 8}" y="${sy(y) + 4}" text-anchor="end">${y}</text>`;
  }
  s += `<text class="axis" x="${(M.l + W - M.r) / 2}" y="${H - 8}" text-anchor="middle">Petal length (cm)</text>
        <text class="axis" transform="translate(14 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle">Petal width (cm)</text>`;
  for (const [x, y, c] of model.points) {
    s += `<circle cx="${sx(x)}" cy="${sy(y)}" r="4" fill="${COLORS[c]}" fill-opacity=".7"/>`;
  }
  s += `<circle id="you" r="9" fill="none" stroke="#1e1b2e" stroke-width="2.5"/>`;
  $("plot").innerHTML = s;
}

function drawBars() {
  $("bars").innerHTML = model.classes.map((c, i) => `
    <div class="bar">
      <div class="bar-head"><span>Iris ${c}</span><span id="pct${i}"></span></div>
      <div class="track"><div class="fill" id="fill${i}" style="background:${COLORS[i]}"></div></div>
    </div>`).join("");
}

function drawControls() {
  values = model.mean.map(v => Math.round(v * 10) / 10);
  LABELS.forEach((label, i) => {
    const [min, max] = model.range[i];
    const field = document.createElement("div");
    field.className = "field";
    field.innerHTML = `<div class="field-head"><label for="f${i}">${label}</label><output for="f${i}"></output></div>
      <input type="range" id="f${i}" min="${min}" max="${max}" step="0.1" value="${values[i]}">`;
    const input = field.querySelector("input"), out = field.querySelector("output");
    const sync = () => { values[i] = +input.value; out.textContent = values[i].toFixed(1) + " cm"; };
    input.addEventListener("input", () => { sync(); update(); });
    sync();
    $("controls").append(field);
  });
}

// Softmax over the logistic regression scores.
function predict() {
  const z = model.coef.map((w, i) => w.reduce((s, wj, j) => s + wj * values[j], model.intercept[i]));
  const top = Math.max(...z);
  const e = z.map(v => Math.exp(v - top));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / sum);
}

function update() {
  const p = predict();
  const best = p.indexOf(Math.max(...p));
  $("top").textContent = `Iris ${model.classes[best]}`;
  $("top").style.color = COLORS[best];
  p.forEach((v, i) => {
    $("pct" + i).textContent = Math.round(v * 100) + "%";
    $("fill" + i).style.width = v * 100 + "%";
  });
  $("you").setAttribute("cx", sx(values[2]));
  $("you").setAttribute("cy", sy(values[3]));
}

async function init() {
  try {
    const res = await fetch("model.json");
    if (!res.ok) throw new Error("model.json not found. Run python train.py, then commit docs/model.json.");
    model = await res.json();
    drawPlot(); drawBars(); drawControls(); update();
    $("note").textContent = `Accuracy on flowers held out from training: ${Math.round(model.accuracy * 100)}%.`;
  } catch (err) {
    $("top").textContent = "No result";
    $("note").textContent = err instanceof TypeError
      ? "Could not load model.json. Open the page through a web server (python -m http.server), not by double-clicking the file."
      : err.message;
  }
}
init();
