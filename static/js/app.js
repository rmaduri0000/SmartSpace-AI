/**
 * SmartSpace AI - Main Application Controller
 * Handles custom user room loading from /room-details,
 * Interactive furniture inventory management (Add, Rotate, Snap, Delete),
 * 2D Canvas & 3D WebGL synchronization, and DQN Layout AI animation.
 */

document.addEventListener('DOMContentLoaded', () => {
  let floorplanCanvas = null;
  let room3D = null;
  let trainingCharts = null;

  let currentLayout = null;
  let currentViewMode = '2d';
  let isOptimizing = false;

  // UI Element References
  const canvasElement = document.getElementById('floorplanCanvas');
  const threejsContainer = document.getElementById('threejsContainer');
  const dqnOptimizeBtn = document.getElementById('dqnOptimizeBtn');
  const resetLayoutBtn = document.getElementById('resetLayoutBtn');
  const exportSpecBtn = document.getElementById('exportSpecBtn');
  const exportPngBtn = document.getElementById('exportPngBtn');
  const actionLog = document.getElementById('actionLog');

  // Metric Displays
  const metricScore = document.getElementById('metricScore');
  const metricCollisions = document.getElementById('metricCollisions');
  const metricCirculation = document.getElementById('metricCirculation');
  const metricCost = document.getElementById('metricCost');
  const budgetBar = document.getElementById('budgetProgressBar');
  const budgetRatioText = document.getElementById('budgetRatioText');
  const ergonomicsStatus = document.getElementById('ergonomicsStatus');

  // Canvas View Controls
  const zoomInBtn = document.getElementById('zoomInBtn');
  const zoomOutBtn = document.getElementById('zoomOutBtn');
  const resetViewBtn = document.getElementById('resetViewBtn');
  const toggleWalkways = document.getElementById('toggleWalkways');
  const toggleClearances = document.getElementById('toggleClearances');

  // Initialize Canvas & 3D Viewer
  floorplanCanvas = new FloorplanCanvas('floorplanCanvas', (newLayout, eventType) => {
    onLayoutManuallyUpdated(newLayout, eventType);
  });

  room3D = new Room3DViewer('threejsContainer');
  trainingCharts = new TrainingCharts('dqnTrainingChart', 'yoloTrainingChart');

  // -------------------------------------------------------------
  // Load Room Data: Check LocalStorage (from /room-details) First!
  // -------------------------------------------------------------
  function initializeRoom() {
    const storedRoom = localStorage.getItem('smartspace_current_room');
    if (storedRoom) {
      try {
        const parsed = JSON.parse(storedRoom);
        currentLayout = parsed;
        updateStudioLayout(currentLayout);
        evaluateLayout(currentLayout);
        updateRoomHeaders(currentLayout);
        if (actionLog) {
          actionLog.innerHTML = `<span class="badge emerald">LOADED</span> Custom room <b>${currentLayout.name || 'Room'}</b> initialized with local models!`;
        }
        return;
      } catch (e) {
        console.error('Failed to parse stored room:', e);
      }
    }
    // Fallback if no custom room created yet
    loadRoomPreset('master_bedroom');
  }

  function updateRoomHeaders(layout) {
    const nameEl = document.getElementById('studioRoomName');
    const metaEl = document.getElementById('studioRoomMeta');
    const w = layout.room_width || layout.dimensions?.width || 4.8;
    const l = layout.room_length || layout.dimensions?.length || 4.0;
    const wFt = (w / 0.3048).toFixed(1);
    const lFt = (l / 0.3048).toFixed(1);
    const budgetVal = layout.budget || 20000;

    if (nameEl) nameEl.textContent = layout.name || 'Custom Studio Room';
    if (metaEl) {
      metaEl.textContent = `Dimensions: ${lFt} ft × ${wFt} ft (${l.toFixed(2)}m × ${w.toFixed(2)}m) · Budget: ₹${budgetVal.toLocaleString()} · Style: ${layout.style || 'Modern'}`;
    }
  }

  async function loadRoomPreset(presetId) {
    try {
      const res = await fetch('/api/sample-rooms');
      const data = await res.json();
      if (data.success && data.rooms[presetId]) {
        currentLayout = data.rooms[presetId];
        updateStudioLayout(currentLayout);
        evaluateLayout(currentLayout);
        updateRoomHeaders(currentLayout);
        if (actionLog) {
          actionLog.innerHTML = `<span class="badge cyan">PRESET</span> Loaded <b>${currentLayout.name}</b>`;
        }
      }
    } catch (err) {
      console.error('Failed to load preset:', err);
    }
  }

  function updateStudioLayout(layout) {
    currentLayout = layout;
    floorplanCanvas.setLayout(layout);
    if (room3D && currentViewMode === '3d') {
      room3D.updateLayout(layout);
    }
    updateInventoryList(layout.furniture || layout.initial_furniture || []);
  }

  function updateMetricsUI(metrics) {
    if (!metrics) return;

    const score = metrics.ergonomics_score || 0;
    if (metricScore) {
      metricScore.textContent = `${score}%`;
      metricScore.className = score >= 80 ? 'metric-value emerald' : (score >= 50 ? 'metric-value cyan' : 'metric-value amber');
    }

    if (ergonomicsStatus) {
      if (score >= 90) ergonomicsStatus.textContent = '🌟 Optimal Ergonomic Layout';
      else if (score >= 70) ergonomicsStatus.textContent = '✅ Good Circulation & Clearance';
      else if (metrics.collision_count > 0) ergonomicsStatus.textContent = '⚠️ Furniture Intersections Detected';
      else ergonomicsStatus.textContent = '⚡ Needs Circulation Optimization';
    }

    if (metricCollisions) {
      metricCollisions.textContent = metrics.collision_count || 0;
      metricCollisions.className = (metrics.collision_count > 0) ? 'metric-value amber' : 'metric-value emerald';
    }

    if (metricCirculation) {
      const ratio = metrics.circulation_ratio !== undefined ? (metrics.circulation_ratio * 100).toFixed(0) : 100;
      metricCirculation.textContent = `${ratio}%`;
    }

    const cost = metrics.total_cost || 0;
    const maxBudget = currentLayout?.budget || 20000;
    if (metricCost) {
      metricCost.textContent = `₹${cost.toLocaleString()}`;
    }

    if (budgetBar) {
      const pct = Math.min(100, (cost / Math.max(1, maxBudget)) * 100);
      budgetBar.style.width = `${pct}%`;
      budgetBar.style.backgroundColor = cost > maxBudget ? '#ef4444' : '#10b981';
    }

    if (budgetRatioText) {
      budgetRatioText.textContent = `₹${cost.toLocaleString()} / ₹${maxBudget.toLocaleString()}`;
    }
  }

  async function evaluateLayout(layout) {
    try {
      const res = await fetch('/api/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(layout)
      });
      const data = await res.json();
      if (data.success && data.metrics) {
        updateMetricsUI(data.metrics);
        if (data.metrics.paths) {
          floorplanCanvas.setPaths(data.metrics.paths);
        }
      }
    } catch (err) {
      console.error('Failed to evaluate layout:', err);
    }
  }

  function onLayoutManuallyUpdated(newLayout, eventType) {
    currentLayout = newLayout;
    if (room3D && currentViewMode === '3d') {
      room3D.updateLayout(currentLayout);
    }
    evaluateLayout(currentLayout);
    updateInventoryList(currentLayout.furniture);
  }

  // -------------------------------------------------------------
  // Furniture Inventory Management (Rotate, Snap, Delete)
  // -------------------------------------------------------------
  function updateInventoryList(furniture) {
    const listEl = document.getElementById('inventoryList');
    if (!listEl) return;
    listEl.innerHTML = '';

    const selectedId = floorplanCanvas.selectedItem?.id;

    furniture.forEach((item, idx) => {
      const div = document.createElement('div');
      div.className = `inventory-item ${selectedId === item.id ? 'active' : ''}`;
      const color = floorplanCanvas.colorMap[item.type] || '#3b82f6';
      
      div.innerHTML = `
        <div class="item-badge" style="cursor: pointer;">
          <span class="color-dot" style="background-color: ${color}"></span>
          <span><b>${item.label || item.type.toUpperCase()}</b></span>
        </div>
        <div class="item-actions">
          <span class="metric-sub">${(item.width || 1).toFixed(1)}x${(item.depth || 1).toFixed(1)}m</span>
          <button class="icon-btn rotate-btn" data-id="${item.id}" title="Rotate 90°">↻</button>
          <button class="icon-btn snap-btn" data-id="${item.id}" title="Snap to Wall">⇄</button>
          <button class="icon-btn delete-btn" data-id="${item.id}" title="Remove Item" style="color: #f43f5e;">✕</button>
        </div>
      `;

      div.querySelector('.item-badge').addEventListener('click', () => {
        floorplanCanvas.selectedItem = item;
        floorplanCanvas.render();
        updateInventoryList(furniture);
      });

      listEl.appendChild(div);
    });

    // Wire up buttons
    document.querySelectorAll('.rotate-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const id = btn.getAttribute('data-id');
        floorplanCanvas.rotateFurniture(id, 90);
      });
    });

    document.querySelectorAll('.snap-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const id = btn.getAttribute('data-id');
        floorplanCanvas.snapToNearestWall(id);
      });
    });

    document.querySelectorAll('.delete-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const id = btn.getAttribute('data-id');
        floorplanCanvas.removeFurniture(id);
      });
    });
  }

  // Add Item From Catalog
  document.querySelectorAll('.catalog-add-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const type = btn.getAttribute('data-type');
      const w = parseFloat(btn.getAttribute('data-w') || 1.2);
      const d = parseFloat(btn.getAttribute('data-d') || 0.8);
      const cost = parseFloat(btn.getAttribute('data-cost') || 2500);
      const label = btn.getAttribute('data-label') || type.toUpperCase();

      floorplanCanvas.addFurniture({
        type: type,
        width: w,
        depth: d,
        cost: cost,
        label: label,
        preferred_wall: ['bed', 'wardrobe', 'desk', 'tv', 'cabinet'].includes(type)
      });

      if (actionLog) {
        actionLog.innerHTML = `<span class="badge cyan">ADDED</span> Added <b>${label}</b> to room!`;
      }
    });
  });

  // -------------------------------------------------------------
  // Viewport Controls & Canvas Tools
  // -------------------------------------------------------------
  if (zoomInBtn) zoomInBtn.addEventListener('click', () => floorplanCanvas.zoomIn());
  if (zoomOutBtn) zoomOutBtn.addEventListener('click', () => floorplanCanvas.zoomOut());
  if (resetViewBtn) resetViewBtn.addEventListener('click', () => floorplanCanvas.resetView());

  if (toggleWalkways) {
    toggleWalkways.addEventListener('change', (e) => {
      floorplanCanvas.showWalkways = e.target.checked;
      floorplanCanvas.render();
    });
  }

  if (toggleClearances) {
    toggleClearances.addEventListener('change', (e) => {
      floorplanCanvas.showClearances = e.target.checked;
      floorplanCanvas.render();
    });
  }

  // 2D / 3D View Mode Toggle
  document.querySelectorAll('.view-mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const mode = btn.getAttribute('data-mode');
      document.querySelectorAll('.view-mode-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      currentViewMode = mode;
      if (mode === '2d') {
        canvasElement.style.display = 'block';
        threejsContainer.style.display = 'none';
        floorplanCanvas.resizeCanvas();
      } else {
        canvasElement.style.display = 'none';
        threejsContainer.style.display = 'block';
        if (room3D && currentLayout) {
          room3D.onResize();
          room3D.updateLayout(currentLayout);
        }
      }
    });
  });

  // -------------------------------------------------------------
  // DQN Layout Optimization Animation
  // -------------------------------------------------------------
  if (dqnOptimizeBtn) {
    dqnOptimizeBtn.addEventListener('click', async () => {
      if (isOptimizing || !currentLayout) return;
      isOptimizing = true;
      dqnOptimizeBtn.disabled = true;
      dqnOptimizeBtn.innerHTML = '<span>⚡ Local DQN AI Thinking...</span>';

      try {
        const res = await fetch('/api/optimize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(currentLayout)
        });
        const data = await res.json();
        
        if (data.success && data.trajectory) {
          await playTrajectoryAnimation(data.trajectory);
          updateStudioLayout(data.final_layout);
          updateMetricsUI(data.final_layout.metrics);
        }
      } catch (err) {
        console.error('DQN optimization failed:', err);
      } finally {
        isOptimizing = false;
        dqnOptimizeBtn.disabled = false;
        dqnOptimizeBtn.innerHTML = '<span>⚡ Run DQN Layout AI</span>';
      }
    });
  }

  async function playTrajectoryAnimation(trajectory) {
    for (let frame of trajectory) {
      updateStudioLayout(frame.layout);
      updateMetricsUI(frame.layout.metrics);
      if (actionLog) {
        actionLog.innerHTML = `<span class="badge purple">STEP ${frame.step}</span> ${frame.action_desc} (Score: <b>${frame.score}%</b>)`;
      }
      await new Promise(r => setTimeout(r, 85));
    }
  }

  // Reset Layout
  if (resetLayoutBtn) {
    resetLayoutBtn.addEventListener('click', () => {
      initializeRoom();
    });
  }

  // Export Specification (.txt)
  if (exportSpecBtn) {
    exportSpecBtn.addEventListener('click', async () => {
      if (!currentLayout) return;
      try {
        const res = await fetch('/api/export-spec', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ layout: currentLayout })
        });
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${(currentLayout.name || 'room').toLowerCase().replace(/\s+/g, '_')}_spec.txt`;
        document.body.appendChild(a);
        a.click();
        a.remove();
      } catch (err) {
        console.error('Export failed:', err);
      }
    });
  }

  // Export Floorplan Snapshot (.png)
  if (exportPngBtn) {
    exportPngBtn.addEventListener('click', () => {
      const dataUrl = floorplanCanvas.exportImage();
      const a = document.createElement('a');
      a.href = dataUrl;
      a.download = `${(currentLayout.name || 'room').toLowerCase().replace(/\s+/g, '_')}_floorplan.png`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    });
  }

  // Initialize
  initializeRoom();
});
