/**
 * SmartSpace AI - Training Analytics & Research Charts
 * Renders DQN learning curves, reward telemetry, and YOLO training loss.
 */

class TrainingCharts {
  constructor(dqnCanvasId, yoloCanvasId) {
    this.dqnCanvas = document.getElementById(dqnCanvasId);
    this.yoloCanvas = document.getElementById(yoloCanvasId);
    this.dqnChart = null;
    this.yoloChart = null;
  }

  renderDQNChart(data) {
    if (!this.dqnCanvas || !window.Chart) return;
    const ctx = this.dqnCanvas.getContext('2d');
    
    if (this.dqnChart) {
      this.dqnChart.destroy();
    }

    const episodes = data.episodes || [];
    const rewards = data.rewards || [];
    const scores = data.ergonomics_scores || [];

    this.dqnChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: episodes,
        datasets: [
          {
            label: 'Ergonomics Score (%)',
            data: scores,
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            fill: true,
            tension: 0.3,
            borderWidth: 2,
            yAxisID: 'y'
          },
          {
            label: 'Episode Reward',
            data: rewards,
            borderColor: '#8b5cf6',
            backgroundColor: 'transparent',
            borderDash: [4, 4],
            tension: 0.2,
            borderWidth: 2,
            yAxisID: 'y1'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Training Episodes', color: '#64748b' },
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8' }
          },
          y: {
            type: 'linear',
            display: true,
            position: 'left',
            title: { display: true, text: 'Score %', color: '#10b981' },
            min: 0,
            max: 100,
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8' }
          },
          y1: {
            type: 'linear',
            display: true,
            position: 'right',
            title: { display: true, text: 'Total Reward', color: '#8b5cf6' },
            grid: { drawOnChartArea: false },
            ticks: { color: '#94a3b8' }
          }
        }
      }
    });
  }

  renderYOLOChart(data) {
    if (!this.yoloCanvas || !window.Chart) return;
    const ctx = this.yoloCanvas.getContext('2d');

    if (this.yoloChart) {
      this.yoloChart.destroy();
    }

    const history = data.history || [];
    const epochs = history.map(h => h.epoch);
    const boxLoss = history.map(h => h.box_loss);
    const map50 = history.map(h => (h.mAP50 * 100).toFixed(1));

    this.yoloChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: epochs,
        datasets: [
          {
            label: 'Validation mAP@50 (%)',
            data: map50,
            borderColor: '#06b6d4',
            backgroundColor: 'rgba(6, 182, 212, 0.1)',
            fill: true,
            tension: 0.3,
            borderWidth: 2,
            yAxisID: 'y'
          },
          {
            label: 'Box Training Loss',
            data: boxLoss,
            borderColor: '#f43f5e',
            backgroundColor: 'transparent',
            tension: 0.2,
            borderWidth: 2,
            yAxisID: 'y1'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Epochs (From Scratch)', color: '#64748b' },
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8' }
          },
          y: {
            type: 'linear',
            display: true,
            position: 'left',
            title: { display: true, text: 'mAP@50 %', color: '#06b6d4' },
            min: 0,
            max: 100,
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8' }
          },
          y1: {
            type: 'linear',
            display: true,
            position: 'right',
            title: { display: true, text: 'Loss', color: '#f43f5e' },
            grid: { drawOnChartArea: false },
            ticks: { color: '#94a3b8' }
          }
        }
      }
    });
  }
}

window.TrainingCharts = TrainingCharts;
