// Scores one application in the browser from scorecard.json: the same rules as src/crs/applicant.py
// (tests/test_page.py runs both on the same applicants and checks they agree).
(function (root) {
  "use strict";

  function missing(v) {
    return v === null || v === undefined || v === "" || (typeof v === "number" && !isFinite(v));
  }

  // Numerical bins are [split_i, split_i+1); the last code is the missing bin.
  // An unseen category goes to the bin holding "OTHER", or to the missing bin if there is none.
  function binCode(c, value) {
    if (c.type === "numerical") {
      const n = c.splits.length + 1;
      if (missing(value)) return n;
      const x = Number(value);
      let code = 0;
      while (code < c.splits.length && x >= c.splits[code]) code++;
      return code;
    }
    const n = c.groups.length;
    if (missing(value)) return n;
    const v = c.keep.includes(value) ? value : "OTHER";
    for (let i = 0; i < c.groups.length; i++) if (c.groups[i].includes(v)) return i;
    for (let i = 0; i < c.groups.length; i++) if (c.groups[i].includes("OTHER")) return i;
    return n;
  }

  function characteristics(app) {
    const amount = app.loan_amount, income = app.annual_income;
    return {
      fico: app.fico_score,
      annual_inc: income,
      dti: app.debt_to_income,
      loan_to_income: !missing(amount) && !missing(income) && Number(income) > 0 ? Number(amount) / Number(income) : null,
      mort_acc: app.mortgage_accounts,
      home_ownership: app.home_ownership,
      inq_last_6mths: app.inquiries_last_6_months,
      revol_util: app.revolving_utilisation,
      purpose: app.purpose,
    };
  }

  function annuityFactor(rate, term) {
    const m = rate / 12;
    return m === 0 ? 1 / term : m / (1 - Math.pow(1 + m, -term));
  }

  function breakEvenRate(pd, card) {
    if (pd >= 1) return null;
    const needed = (pd * card.loss_rate / (1 - pd) + 1) / (1 - card.fee);
    let low = 0, high = 1;
    if (annuityFactor(high, card.term) * card.term < needed) return null;
    for (let i = 0; i < 60; i++) {
      const mid = (low + high) / 2;
      if (annuityFactor(mid, card.term) * card.term < needed) low = mid; else high = mid;
    }
    return high;
  }

  function monthlyPayment(amount, rate, term) {
    const m = rate / 12;
    return amount * m / (1 - Math.pow(1 + m, -term));
  }

  // The lending rules that run before the score (src/crs/policy.py). A missing value breaks no rule.
  function checkRules(card, app, rate) {
    const p = card.policy, values = Object.assign({}, app);
    const r = Math.max(rate, p.assumed_rate);
    values.loan_to_income = null;
    values.dti_with_loan = null;
    if (!missing(app.loan_amount) && !missing(app.annual_income) && Number(app.annual_income) > 0) {
      values.loan_to_income = Number(app.loan_amount) / Number(app.annual_income);
      if (!missing(app.debt_to_income)) {
        values.dti_with_loan = Number(app.debt_to_income) +
          100 * monthlyPayment(Number(app.loan_amount), r, card.term) / (Number(app.annual_income) / 12);
      }
    }
    const broken = [];
    Object.keys(p.required).forEach(function (field) {
      if (missing(app[field])) broken.push({ rule: "required", text: p.required[field], value: null, limit: null });
    });
    p.rules.forEach(function (rule) {
      const v = values[rule.field];
      if (missing(v)) return;
      const x = Number(v);
      if ((rule.how === "below" && x < rule.limit) || (rule.how === "above" && x > rule.limit)) {
        broken.push({ rule: rule.name, text: rule.text, value: Math.round(x * 1e4) / 1e4, limit: rule.limit });
      }
    });
    return broken;
  }

  function score(card, app) {
    const x = characteristics(app);
    const lines = card.characteristics.map(function (c) {
      const code = binCode(c, x[c.name]);
      return {
        characteristic: c.name,
        description: c.description,
        bin: c.labels[code],
        points: c.points[code],
        max_points: Math.max.apply(null, c.points),
      };
    });
    const total = lines.reduce(function (s, l) { return s + l.points; }, 0);
    const pd = 1 / (1 + Math.exp((total - card.offset) / card.factor));
    const rate = breakEvenRate(pd, card);
    // Same order as Python: most points lost first; ties keep the scorecard's order.
    const reasons = lines
      .map(function (l, i) { return { l: l, i: i, gap: l.max_points - l.points }; })
      .filter(function (r) { return r.gap > 0; })
      .sort(function (a, b) { return b.gap - a.gap || a.i - b.i; })
      .slice(0, 3)
      .map(function (r) {
        return { characteristic: r.l.characteristic, text: r.l.description + ": " + r.l.bin, points_lost: r.gap };
      });
    let decision = "approve", rule = "rate covers the expected loss";
    const broken = checkRules(card, app, rate === null ? 1 : rate);
    if (broken.length) { decision = "decline"; rule = "lending rule"; }
    else if (pd > card.max_pd) { decision = "decline"; rule = "pd above policy limit"; }
    else if (rate === null || rate > card.max_rate) { decision = "decline"; rule = "no rate covers the expected loss"; }
    return {
      score: total,
      pd: Math.round(pd * 1e4) / 1e4,
      break_even_rate: rate === null ? null : Math.round(rate * 1e4) / 1e4,
      decision: decision,
      decision_rule: rule,
      rules_broken: broken,
      reasons: reasons,
      points: lines,
    };
  }

  const api = { score: score, binCode: binCode, breakEvenRate: breakEvenRate };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Scorer = api;
})(this);
