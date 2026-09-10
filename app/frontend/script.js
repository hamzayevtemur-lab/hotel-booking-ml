/* ============================================================
   Hotel Booking Cancellation Predictor — Frontend JS
   ============================================================ */

// ── Smart Date & Stay Auto-Calculator ─────────────────────────
const MONTH_NAMES = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December'
];

function getISOWeekNumber(d) {
  const date = new Date(Date.UTC(d.getFullYear(), d.getMonth(), d.getDate()));
  const dayNum = date.getUTCDay() || 7;
  date.setUTCDate(date.getUTCDate() + 4 - dayNum);
  const yearStart = new Date(Date.UTC(date.getUTCFullYear(), 0, 1));
  return Math.ceil((((date - yearStart) / 86400000) + 1) / 7);
}

function calculateLeadTime(arrivalDate) {
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const arrival = new Date(arrivalDate.getFullYear(), arrivalDate.getMonth(), arrivalDate.getDate());
  const diffTime = arrival - today;
  const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
  return Math.max(0, diffDays);
}

function calculateNights(arrivalDate, totalNights) {
  let weekdayNights = 0;
  let weekendNights = 0;
  for (let i = 0; i < totalNights; i++) {
    const curDate = new Date(arrivalDate.getFullYear(), arrivalDate.getMonth(), arrivalDate.getDate() + i);
    const day = curDate.getDay(); // 0 = Sun, 6 = Sat
    if (day === 0 || day === 6) {
      weekendNights++;
    } else {
      weekdayNights++;
    }
  }
  return { weekdayNights, weekendNights };
}

let isAutoUpdating = false;

function autoFillFromDate(dateObj) {
  if (isAutoUpdating || isNaN(dateObj.getTime())) return;
  isAutoUpdating = true;

  const year  = dateObj.getFullYear();
  const month = MONTH_NAMES[dateObj.getMonth()];
  const day   = dateObj.getDate();

  // 1. Sync Date Picker input value
  const datePicker = document.getElementById('arrival_date_picker');
  if (datePicker) {
    const yyyy = year;
    const mm   = String(dateObj.getMonth() + 1).padStart(2, '0');
    const dd   = String(day).padStart(2, '0');
    datePicker.value = `${yyyy}-${mm}-${dd}`;
  }

  // 2. Sync Year select
  const yearSelect = document.getElementById('arrival_date_year');
  if (yearSelect) {
    if (![...yearSelect.options].some(o => o.value == year)) {
      const opt = document.createElement('option');
      opt.value = year; opt.textContent = year;
      yearSelect.appendChild(opt);
    }
    yearSelect.value = year;
  }

  // 3. Sync Month & Day
  const monthSelect = document.getElementById('arrival_date_month');
  if (monthSelect) monthSelect.value = month;

  const dayInput = document.getElementById('arrival_date_day_of_month');
  if (dayInput) dayInput.value = day;

  // 4. Auto-calculate Week Number
  const weekInput = document.getElementById('arrival_date_week_number');
  if (weekInput) weekInput.value = getISOWeekNumber(dateObj);

  // 5. Auto-calculate Lead Time (days from today)
  const leadInput = document.getElementById('lead_time');
  if (leadInput) leadInput.value = calculateLeadTime(dateObj);

  // 6. Auto-calculate Weekday & Weekend Nights
  const totalNightsEl = document.getElementById('total_nights');
  const totalNights   = Math.max(1, Number(totalNightsEl ? totalNightsEl.value : 3));
  const { weekdayNights, weekendNights } = calculateNights(dateObj, totalNights);

  const weekdayEl = document.getElementById('stays_in_week_nights');
  const weekendEl = document.getElementById('stays_in_weekend_nights');
  if (weekdayEl) weekdayEl.value = weekdayNights;
  if (weekendEl) weekendEl.value = weekendNights;

  isAutoUpdating = false;
}

function getSelectedDate() {
  const y = Number(document.getElementById('arrival_date_year').value);
  const mStr = document.getElementById('arrival_date_month').value;
  const m = MONTH_NAMES.indexOf(mStr);
  const d = Math.max(1, Math.min(31, Number(document.getElementById('arrival_date_day_of_month').value)));
  return new Date(y, m, d);
}

// Event Listeners for Date & Stay Auto-Calculation
document.addEventListener('DOMContentLoaded', () => {
  const datePicker  = document.getElementById('arrival_date_picker');
  const yearSelect  = document.getElementById('arrival_date_year');
  const monthSelect = document.getElementById('arrival_date_month');
  const dayInput    = document.getElementById('arrival_date_day_of_month');
  const totalNights = document.getElementById('total_nights');
  const weekNights  = document.getElementById('stays_in_week_nights');
  const wkndNights  = document.getElementById('stays_in_weekend_nights');

  // Set initial default date (Oct 10, 2026)
  const initialDate = new Date(2026, 9, 10);
  autoFillFromDate(initialDate);

  // 1. Date Picker changed
  datePicker?.addEventListener('change', () => {
    if (!datePicker.value) return;
    const [y, m, d] = datePicker.value.split('-').map(Number);
    autoFillFromDate(new Date(y, m - 1, d));
  });

  // 2. Year, Month, Day changed directly
  [yearSelect, monthSelect, dayInput].forEach(el => {
    el?.addEventListener('change', () => autoFillFromDate(getSelectedDate()));
    el?.addEventListener('input',  () => autoFillFromDate(getSelectedDate()));
  });

  // 3. Total Nights changed -> update weekday & weekend breakdown
  totalNights?.addEventListener('input', () => autoFillFromDate(getSelectedDate()));

  // 4. Manual change to Weekday or Weekend Nights -> update Total Nights
  [weekNights, wkndNights].forEach(el => {
    el?.addEventListener('input', () => {
      if (isAutoUpdating) return;
      const total = Number(weekNights.value || 0) + Number(wkndNights.value || 0);
      if (totalNights) totalNights.value = Math.max(1, total);
    });
  });
});


// ── Tab navigation ───────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.tab;

    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));

    btn.classList.add('active');
    document.getElementById(`tab-${target}`).classList.add('active');
  });
});


// ── Gauge animation ──────────────────────────────────────────
const CIRCUMFERENCE = 2 * Math.PI * 54; // 339.3

function animateGauge(probability) {
  const fill    = document.getElementById('gauge-fill');
  const pctEl   = document.getElementById('gauge-pct');
  const target  = probability * 100;
  const offset  = CIRCUMFERENCE * (1 - probability);

  // Color thresholds
  let color;
  if (probability < 0.30)      color = '#10b981'; // green
  else if (probability < 0.55) color = '#f59e0b'; // amber
  else                          color = '#ef4444'; // red

  fill.style.stroke             = color;
  fill.style.strokeDashoffset   = offset;

  // Counter animation
  let current = 0;
  const step  = target / 60;
  const timer = setInterval(() => {
    current = Math.min(current + step, target);
    pctEl.textContent = Math.round(current) + '%';
    if (current >= target) clearInterval(timer);
  }, 16);
}


// ── Build payload from form ───────────────────────────────────
function buildPayload() {
  const get  = id => document.getElementById(id);
  const num  = id => Number(get(id).value);
  const str  = id => get(id).value.trim();

  return {
    hotel:                          str('hotel'),
    lead_time:                      num('lead_time'),
    arrival_date_year:              num('arrival_date_year'),
    arrival_date_month:             str('arrival_date_month'),
    arrival_date_week_number:       num('arrival_date_week_number'),
    arrival_date_day_of_month:      num('arrival_date_day_of_month'),
    stays_in_weekend_nights:        num('stays_in_weekend_nights'),
    stays_in_week_nights:           num('stays_in_week_nights'),
    adults:                         num('adults'),
    children:                       num('children'),
    babies:                         num('babies'),
    meal:                           str('meal'),
    country:                        str('country').toUpperCase(),
    market_segment:                 str('market_segment'),
    distribution_channel:           str('distribution_channel'),
    is_repeated_guest:              get('is_repeated_guest').checked ? 1 : 0,
    previous_cancellations:         num('previous_cancellations'),
    previous_bookings_not_canceled: num('previous_bookings_not_canceled'),
    reserved_room_type:             str('reserved_room_type'),
    assigned_room_type:             str('assigned_room_type'),
    deposit_type:                   str('deposit_type'),
    agent:                          'Unknown',
    company:                        'Unknown',
    days_in_waiting_list:           num('days_in_waiting_list'),
    customer_type:                  str('customer_type'),
    adr:                            num('adr'),
    required_car_parking_spaces:    num('required_car_parking_spaces'),
    total_of_special_requests:      num('total_of_special_requests'),
  };
}


// ── Render prediction result ──────────────────────────────────
function renderResult(data) {
  const panel     = document.getElementById('result-panel');
  const badge     = document.getElementById('result-badge');
  const statProb  = document.getElementById('stat-prob');
  const statRisk  = document.getElementById('stat-risk');
  const note      = document.getElementById('result-note');

  // Unhide panel
  panel.classList.remove('hidden');
  panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  // Badge
  if (data.prediction === 1) {
    badge.textContent  = '⚠️  LIKELY CANCELLED';
    badge.className    = 'result-badge cancel';
  } else {
    badge.textContent  = '✅  BOOKING SAFE';
    badge.className    = 'result-badge safe';
  }

  // Stats
  statProb.textContent = (data.probability * 100).toFixed(1) + '%';

  const riskLower = data.risk_level.toLowerCase();
  statRisk.innerHTML = `<span class="risk-pill ${riskLower}">${data.risk_level}</span>`;

  // Note
  const notes = {
    Low:    'This booking has a low cancellation probability. Standard monitoring is sufficient.',
    Medium: 'This booking has a moderate cancellation risk. Consider a follow-up confirmation.',
    High:   'High cancellation risk detected. Consider proactive overbooking or a non-refundable deposit policy.',
  };
  note.textContent = notes[data.risk_level] || '';

  // Gauge
  animateGauge(data.probability);
}


// ── Form submission ───────────────────────────────────────────
document.getElementById('prediction-form').addEventListener('submit', async e => {
  e.preventDefault();

  const btn     = document.getElementById('submit-btn');
  const spinner = document.getElementById('spinner');
  const btnText = document.getElementById('btn-text');

  btn.disabled       = true;
  spinner.style.display = 'block';
  btnText.textContent   = 'Analysing…';

  try {
    const payload  = buildPayload();
    const response = await fetch('/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Prediction failed.');
    }

    const data = await response.json();
    renderResult(data);

  } catch (err) {
    alert('Error: ' + err.message);
    console.error(err);
  } finally {
    btn.disabled          = false;
    spinner.style.display = 'none';
    btnText.textContent   = '⚡ Analyse Booking';
  }
});
