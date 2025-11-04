const ctx = document.getElementById("demandChart").getContext("2d");
let chart;
let predictionLabels = [];
let predictionValues = [];
let modelAccuracy = null;

// Handle form submission
document.getElementById("predictForm").addEventListener("submit", async function (event) {
  event.preventDefault();

  const requestData = {
    store_id: parseInt(document.getElementById("store_id").value),
    sku_id: parseInt(document.getElementById("sku_id").value),
    total_price: parseFloat(document.getElementById("total_price").value),
    base_price: parseFloat(document.getElementById("base_price").value),
    is_featured_sku: parseInt(document.getElementById("is_featured_sku").value),
    is_display_sku: parseInt(document.getElementById("is_display_sku").value),
  };

  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestData)
    });

    const data = await response.json();

    if (response.ok) {
      // Fill result card
      document.getElementById("units").textContent = data.predicted_units;
      document.getElementById("percentage").textContent = data.predicted_demand_percentage;
      document.getElementById("demandHigh").textContent = data.is_demand_high;
      document.getElementById("resultCard").classList.remove("d-none");

      // Fill R² score if available
      if (modelAccuracy !== null) {
        document.getElementById("r2Display").textContent = modelAccuracy;
      }

      // Update chart
      predictionLabels.push(`P${predictionLabels.length + 1}`);
      predictionValues.push(data.predicted_demand_percentage);
      updateChart();

    } else {
      alert(data.error || "⚠️ Prediction failed.");
    }

  } catch (error) {
    console.error("Request failed:", error);
    alert("⚠️ Cannot connect to the backend.");
  }
});

// Load model R² accuracy and fill both top banner + result card
fetch("http://127.0.0.1:5000/accuracy")
  .then(res => res.json())
  .then(data => {
    if (data.r2_score !== undefined) {
      modelAccuracy = data.r2_score;
      document.getElementById("accuracyVal").textContent = modelAccuracy;
    } else {
      document.getElementById("accuracyVal").textContent = "Unavailable";
    }
  })
  .catch(err => {
    console.error("Failed to fetch accuracy:", err);
    document.getElementById("accuracyVal").textContent = "Error";
  });

// Create or update the demand chart
function updateChart() {
  if (!chart) {
    chart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: predictionLabels,
        datasets: [{
          label: "Predicted Demand (%)",
          data: predictionValues,
          backgroundColor: "#007bff"
        }]
      },
      options: {
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            title: { display: true, text: "Demand %" }
          }
        }
      }
    });
  } else {
    chart.data.labels = predictionLabels;
    chart.data.datasets[0].data = predictionValues;
    chart.update();
  }
}
