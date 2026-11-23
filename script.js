const qualityInput = document.querySelector('#quality');
const qualityValue = document.querySelector('#quality-value');
const form = document.querySelector('#valuation-form');
const resultLabel = document.querySelector('.result>span');
const resultPrice = document.querySelector('#price');
const resultNote = document.querySelector('.result>small');
let model = null;

qualityInput.addEventListener('input', () => {
  qualityValue.textContent = `${qualityInput.value} / 10`;
});

async function loadModel() {
  try {
    const response = await fetch('./model/ridge_model.json', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    model = await response.json();
    resultLabel.textContent = `RIDGE · ${model.model_version} · USD`;
    resultNote.textContent = `CV MAE $${Math.round(model.cv_mae_usd_mean).toLocaleString('en-US')} · holdout MAE $${Math.round(model.test_mae).toLocaleString('en-US')}`;
  } catch (error) {
    resultLabel.textContent = 'MODEL CHƯA ĐƯỢC TẢI';
    resultNote.textContent = 'Kiểm tra model/ridge_model.json và mở trang qua HTTP server.';
  }
}

form.addEventListener('submit', (event) => {
  event.preventDefault();
  if (!model) {
    resultLabel.textContent = 'MODEL CHƯA SẴN SÀNG';
    return;
  }

  const values = {
    gr_liv_area: Number(document.querySelector('#area').value),
    overall_qual: Number(qualityInput.value),
    bedroom_abvgr: Number(document.querySelector('#beds').value),
    full_bath: Number(document.querySelector('#baths').value),
    year_built: Number(document.querySelector('#year-built').value),
  };
  if (Object.values(values).some((value) => !Number.isFinite(value))) return;

  let logPrice = model.intercept;
  model.features.forEach((feature, index) => {
    const value = values[feature.name];
    const standardized = (value - feature.mean) / feature.scale;
    logPrice += model.coefficients[index] * standardized;
  });
  const predictedPrice = Math.max(0, Math.expm1(logPrice));
  resultLabel.textContent = 'ƯỚC LƯỢNG GIÁ GIAO DỊCH LỊCH SỬ · USD';
  resultPrice.textContent = new Intl.NumberFormat('en-US', {
    style: 'currency', currency: 'USD', maximumFractionDigits: 0,
  }).format(predictedPrice);
  resultNote.textContent = `${model.dataset} · Ridge log1p · không phải báo giá hiện tại.`;
});

loadModel();

async function loadComparison() {
  const body = document.querySelector('#research-results');
  try {
    const response = await fetch('./model/research_results.json', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (!Array.isArray(data.models) || data.models.length === 0) throw new Error('Invalid results schema');
    body.replaceChildren();
    const money = (value) => `$${Math.round(value).toLocaleString('en-US')}`;
    data.models.forEach((item) => {
      const row = document.createElement('tr');
      [item.model,
        `${money(item.cv_mae)} ± ${money(item.cv_mae_std)}`,
        money(item.test_mae), money(item.test_rmse), item.test_r2.toFixed(3)]
        .forEach((value) => {
          const cell = document.createElement('td');
          cell.textContent = value;
          row.append(cell);
        });
      body.append(row);
    });
    document.querySelector('#results-note').textContent =
      `Nguồn: metrics JSON của nhóm · ${data.scope} · holdout ${data.holdout_rows} dòng.`;
  } catch (error) {
    body.replaceChildren();
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 5;
    cell.textContent = 'Chưa tải được model/research_results.json. Kiểm tra file JSON và mở trang qua HTTP server.';
    row.append(cell);
    body.append(row);
  }
}

loadComparison();
