
//const API_BASE = "http://127.0.0.1:8000";

//export async function analyzeFeatures(features) {
  //const res = await fetch(`${API_BASE}/explain`, {
    //method: "POST",
    //headers: {
      //"Content-Type": "application/json",
    //},
    //body: JSON.stringify({
      //features: features,
    //}),
  //});

  //if (!res.ok) {
    //const text = await res.text();
    //throw new Error(text || "Backend request failed");
  //}

  //return res.json();
//}


//const BASE_URL = "http://127.0.0.1:8000";

//export async function analyzeFeatures(features) {
  //const response = await fetch(`${BASE_URL}/explain`, {
    //method: "POST",
    //headers: {
      //"Content-Type": "application/json",
    //},
    //body: JSON.stringify({ features }),
  //});

  //if (!response.ok) {
    //throw new Error("Backend request failed");
  //}

  //const data = await response.json();
  //return data;
//}


// frontend/src/api.js

//export async function analyzeFeatures(features) {
  //const response = await fetch("http://127.0.0.1:8000/analyze", {
    //method: "POST",
    //headers: {
      //"Content-Type": "application/json",
    //},
    //body: JSON.stringify({
      //features: features,
    //}),
  //});

  //if (!response.ok) {
    //const text = await response.text();
    //throw new Error(text || "Backend request failed");
  //}
  //const data = await response.json();
  //return data;
//}



export async function analyzeFeatures(payload) {
  const res = await fetch("http://localhost:8000/analyze", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    throw new Error("Analyze request failed");
  }

  return res.json();
}
