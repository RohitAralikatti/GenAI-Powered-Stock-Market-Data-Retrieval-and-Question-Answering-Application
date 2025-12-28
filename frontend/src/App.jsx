import { useState, useEffect, useRef } from "react";
import { gsap } from "gsap";
import { analyzeFeatures } from "./api";

function App() {
  // ---------- REFS FOR GSAP ----------
  const heroRef = useRef(null);
  const controlsRef = useRef(null);
  const resultsRef = useRef(null);

  // ---------- USER INPUT (INTENT, NOT FEATURES) ----------
  const [scenario, setScenario] = useState("neutral");
  const [risk, setRisk] = useState(0.5);
  const [horizon, setHorizon] = useState(12);

  // ---------- STATE ----------
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // ---------- GSAP: PAGE ENTRY ----------
  useEffect(() => {
    const tl = gsap.timeline();

    tl.from(heroRef.current, {
      opacity: 0,
      y: -40,
      duration: 0.7,
      ease: "power3.out",
    }).from(
      controlsRef.current,
      {
        opacity: 0,
        y: 20,
        duration: 0.6,
        ease: "power3.out",
      },
      "-=0.3"
    );
  }, []);

  // ---------- GSAP: RESULTS ENTRY ----------
  useEffect(() => {
    if (!result || !resultsRef.current) return;

    gsap.fromTo(
      resultsRef.current.children,
      { opacity: 0, y: 20 },
      {
        opacity: 1,
        y: 0,
        duration: 0.5,
        stagger: 0.15,
        ease: "power2.out",
      }
    );
  }, [result]);

  // ---------- ANALYZE ----------
  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);

    // Button feedback
    gsap.to("button", {
      scale: 0.95,
      duration: 0.1,
      yoyo: true,
      repeat: 1,
    });

    try {
      const res = await analyzeFeatures({
        scenario,
        risk,
        horizon,
      });
      setResult(res);
    } catch (e) {
      setError(e.message || "Backend request failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      {/* ---------- HERO ---------- */}
      <h1 ref={heroRef} style={styles.title}>
        FinTech GenAI Analyzer
      </h1>

      {/* ---------- CONTROLS ---------- */}
      <div ref={controlsRef} style={styles.controls}>
        <div style={styles.controlBlock}>
          <label>Scenario</label>
          <select
            value={scenario}
            onChange={(e) => setScenario(e.target.value)}
          >
            <option value="conservative">Conservative</option>
            <option value="neutral">Neutral</option>
            <option value="aggressive">Aggressive</option>
          </select>
        </div>

        <div style={styles.controlBlock}>
          <label>Risk Level: {risk.toFixed(2)}</label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={risk}
            onChange={(e) => setRisk(Number(e.target.value))}
          />
        </div>

        <div style={styles.controlBlock}>
          <label>Investment Horizon (months)</label>
          <input
            type="number"
            min="1"
            max="60"
            value={horizon}
            onChange={(e) => setHorizon(Number(e.target.value))}
          />
        </div>

        <button onClick={handleAnalyze} disabled={loading}>
          {loading ? "Analyzing..." : "Analyze"}
        </button>

        {error && <p style={styles.error}>{error}</p>}
      </div>

      {/* ---------- RESULTS ---------- */}
      {result && (
        <div ref={resultsRef} style={styles.results}>
          <section>
            <h2>Prediction</h2>
            <p>
              <strong>Class:</strong> {result.prediction}
            </p>
            <p>
              <strong>Probabilities:</strong>{" "}
              {result.probabilities.map((p, i) => (
                <span key={i}> {p.toFixed(3)} </span>
              ))}
            </p>
          </section>

          <section>
            <h2>Top SHAP Features</h2>
            <table style={styles.table}>
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>SHAP Value</th>
                  <th>Impact</th>
                </tr>
              </thead>
              <tbody>
                {result.top_features.map((f, i) => (
                  <tr
                    key={i}
                    style={{
                      background:
                        f.shap_value > 0
                          ? "rgba(0, 200, 0, 0.06)"
                          : "rgba(200, 0, 0, 0.06)",
                    }}
                  >
                    <td>{f.feature}</td>
                    <td>{f.shap_value.toFixed(5)}</td>
                    <td>{f.abs_shap.toFixed(5)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          <section>
            <h2>SHAP Summary</h2>
            <p>{result.shap_summary}</p>
          </section>

          <section>
            <h2>AI Explanation</h2>
            <p>{result.llm_explanation}</p>
          </section>
        </div>
      )}
    </div>
  );
}

// ---------- STYLES ----------
const styles = {
  container: {
    padding: 32,
    maxWidth: 900,
    margin: "0 auto",
    fontFamily: "Inter, system-ui, sans-serif",
    background: "#fafafa",
  },
  title: {
    marginBottom: 24,
  },
  controls: {
    display: "grid",
    gap: 16,
    marginBottom: 32,
  },
  controlBlock: {
    display: "flex",
    flexDirection: "column",
    gap: 6,
  },
  results: {
    display: "grid",
    gap: 32,
    marginTop: 40,
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
  },
  error: {
    color: "red",
  },
};

export default App;
