document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const btnReset = document.getElementById('btn-reset');
    const errorMessage = document.getElementById('error-message');
    const resultCard = document.getElementById('result-card');
    
    // UI Elements for GLCM Results
    const validationStatus = document.getElementById('validation-status');
    const ripenessBadge = document.getElementById('ripeness-badge');
    const valContrast = document.getElementById('val-contrast');
    const valDissimilarity = document.getElementById('val-dissimilarity');
    const valHomogeneity = document.getElementById('val-homogeneity');
    const valEnergy = document.getElementById('val-energy');
    const valCorrelation = document.getElementById('val-correlation');
    const valAsm = document.getElementById('val-asm');

    // UI Elements for HSV Results
    const valMeanH = document.getElementById('val-mean-h');
    const valStdH = document.getElementById('val-std-h');
    const valSkewH = document.getElementById('val-skew-h');
    const valMeanS = document.getElementById('val-mean-s');
    const valStdS = document.getElementById('val-std-s');
    const valSkewS = document.getElementById('val-skew-s');
    const valMeanV = document.getElementById('val-mean-v');
    const valStdV = document.getElementById('val-std-v');
    const valSkewV = document.getElementById('val-skew-v');

    // Drag and Drop Events
    dropzone.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    btnReset.addEventListener('click', () => {
        resetUI();
    });

    function handleFile(file) {
        // Validate file type
        if (!file.type.match('image.*')) {
            showError('Hanya file gambar (JPG/PNG) yang diperbolehkan.');
            return;
        }

        // Validate size (< 5MB)
        if (file.size > 5 * 1024 * 1024) {
            showError('Ukuran file maksimal 5MB.');
            return;
        }

        hideError();
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropzone.style.display = 'none';
            previewContainer.style.display = 'block';
            
            // Start uploading & processing
            uploadAndPredict(file);
        };
        reader.readAsDataURL(file);
    }

    function uploadAndPredict(file) {
        // Setup Loading State
        resultCard.style.display = 'block';
        validationStatus.textContent = 'Memproses Gambar...';
        validationStatus.className = 'status-badge';
        ripenessBadge.textContent = 'Menghitung GLCM & HSV...';
        ripenessBadge.className = 'badge';
        resetMetrics();

        // Hide confidence section during loading
        const confidenceSection = document.getElementById('confidence-section');
        if (confidenceSection) confidenceSection.style.display = 'none';

        const formData = new FormData();
        formData.append('image', file);

        fetch('/api/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json().then(data => ({status: response.status, body: data})))
        .then(result => {
            const data = result.body;
            
            if (result.status !== 200 || !data.valid) {
                // Outlier or Error
                validationStatus.textContent = 'Tidak Valid: ' + (data.error || 'Terjadi Kesalahan');
                validationStatus.className = 'status-badge';
                ripenessBadge.textContent = 'GAGAL';
                ripenessBadge.className = 'badge';
                
                if (data.features) {
                    updateGlcmMetrics(data.features);
                }
                if (data.hsv_features) {
                    updateHsvMetrics(data.hsv_features);
                }

                // Hide live PCA container and destroy chart on error
                const pcaContainer = document.getElementById('result-pca-container');
                if (pcaContainer) {
                    pcaContainer.style.display = 'none';
                }
                if (liveChartInstance) {
                    liveChartInstance.destroy();
                    liveChartInstance = null;
                }
                return;
            }

            // Success Valid
            validationStatus.textContent = 'Valid: Objek Dikenali';
            validationStatus.className = 'status-badge valid';
            
            // Ripeness
            ripenessBadge.textContent = data.label.toUpperCase();
            
            // Apply color class based on label
            const labelLower = data.label.toLowerCase();
            if (labelLower.includes('matang')) {
                ripenessBadge.className = 'badge matang';
            } else if (labelLower.includes('mengkal')) {
                ripenessBadge.className = 'badge mengkal';
            } else if (labelLower.includes('mentah')) {
                ripenessBadge.className = 'badge mentah';
            } else {
                ripenessBadge.className = 'badge';
            }

            // Confidence Score
            if (data.confidence !== undefined && data.probabilities) {
                showConfidence(data.confidence, data.probabilities, data.label);
            }

            // GLCM Metrics
            if (data.features) {
                updateGlcmMetrics(data.features);
            }

            // HSV Metrics
            if (data.hsv_features) {
                updateHsvMetrics(data.hsv_features);
            }

            // Draw Live PCA Chart
            if (data.pca_x !== undefined && data.pca_y !== undefined) {
                const pcaContainer = document.getElementById('result-pca-container');
                if (pcaContainer) {
                    pcaContainer.style.display = 'block';
                }
                drawLivePcaChart(data.pca_x, data.pca_y, data.predicted_label);
            }
        })
        .catch(err => {
            console.error(err);
            validationStatus.textContent = 'Error koneksi ke server';
            ripenessBadge.textContent = 'ERROR';
        });
    }

    function showConfidence(confidence, probabilities, predictedLabel) {
        const section = document.getElementById('confidence-section');
        const bar = document.getElementById('confidence-bar');
        const text = document.getElementById('confidence-text');
        const grid = document.getElementById('probability-grid');

        if (!section || !bar || !text || !grid) return;

        section.style.display = 'block';

        // Animate confidence bar
        const pct = (confidence * 100).toFixed(1);
        bar.style.width = pct + '%';
        text.textContent = pct + '%';

        // Color based on confidence level
        if (confidence >= 0.8) {
            bar.className = 'confidence-bar confidence-high';
        } else if (confidence >= 0.5) {
            bar.className = 'confidence-bar confidence-medium';
        } else {
            bar.className = 'confidence-bar confidence-low';
        }

        // Probability breakdown per class
        grid.innerHTML = '';
        const colorMap = {
            'Matang': 'var(--color-matang)',
            'Mengkal': 'var(--color-mengkal)',
            'Mentah': 'var(--color-mentah)',
        };

        for (const [className, prob] of Object.entries(probabilities)) {
            const item = document.createElement('div');
            item.className = 'probability-item';
            const isActive = className === predictedLabel;
            item.innerHTML = `
                <span class="prob-label" style="${isActive ? 'font-weight:700;' : ''}">${className}</span>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width: ${(prob * 100).toFixed(1)}%; background: ${colorMap[className] || '#94a3b8'};"></div>
                </div>
                <span class="prob-value" style="${isActive ? 'font-weight:700;' : ''}">${(prob * 100).toFixed(1)}%</span>
            `;
            grid.appendChild(item);
        }
    }

    function updateGlcmMetrics(f) {
        valContrast.textContent = f.contrast ? f.contrast.toFixed(4) : '-';
        valDissimilarity.textContent = f.dissimilarity ? f.dissimilarity.toFixed(4) : '-';
        valHomogeneity.textContent = f.homogeneity ? f.homogeneity.toFixed(4) : '-';
        valEnergy.textContent = f.energy ? f.energy.toFixed(4) : '-';
        valCorrelation.textContent = f.correlation ? f.correlation.toFixed(4) : '-';
        valAsm.textContent = f.ASM ? f.ASM.toFixed(4) : '-';
    }

    function updateHsvMetrics(h) {
        valMeanH.textContent = h.mean_h !== undefined ? h.mean_h.toFixed(2) : '-';
        valStdH.textContent = h.std_h !== undefined ? h.std_h.toFixed(2) : '-';
        valSkewH.textContent = h.skew_h !== undefined ? h.skew_h.toFixed(4) : '-';
        valMeanS.textContent = h.mean_s !== undefined ? h.mean_s.toFixed(2) : '-';
        valStdS.textContent = h.std_s !== undefined ? h.std_s.toFixed(2) : '-';
        valSkewS.textContent = h.skew_s !== undefined ? h.skew_s.toFixed(4) : '-';
        valMeanV.textContent = h.mean_v !== undefined ? h.mean_v.toFixed(2) : '-';
        valStdV.textContent = h.std_v !== undefined ? h.std_v.toFixed(2) : '-';
        valSkewV.textContent = h.skew_v !== undefined ? h.skew_v.toFixed(4) : '-';
    }

    function resetMetrics() {
        const glcmMetrics = [valContrast, valDissimilarity, valHomogeneity, valEnergy, valCorrelation, valAsm];
        glcmMetrics.forEach(m => m.textContent = '-');

        const hsvMetrics = [valMeanH, valStdH, valSkewH, valMeanS, valStdS, valSkewS, valMeanV, valStdV, valSkewV];
        hsvMetrics.forEach(m => m.textContent = '-');
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        errorMessage.style.display = 'block';
    }

    function hideError() {
        errorMessage.style.display = 'none';
    }

    function resetUI() {
        fileInput.value = '';
        imagePreview.src = '';
        previewContainer.style.display = 'none';
        dropzone.style.display = 'block';
        resultCard.style.display = 'none';
        hideError();

        // Reset confidence section
        const confidenceSection = document.getElementById('confidence-section');
        if (confidenceSection) confidenceSection.style.display = 'none';

        // Reset PCA Container and Chart Instance
        const pcaContainer = document.getElementById('result-pca-container');
        if (pcaContainer) {
            pcaContainer.style.display = 'none';
        }
        if (liveChartInstance) {
            liveChartInstance.destroy();
            liveChartInstance = null;
        }
    }
});

// Fetch Dashboard Data
document.addEventListener('DOMContentLoaded', () => {
    fetchDashboardData();
});

function fetchDashboardData() {
    fetch('/api/dashboard_data')
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            console.warn(data.error);
            return;
        }
        
        // Update Accuracy Score
        const accEl = document.getElementById('test-accuracy');
        if (accEl && data.test_metrics) {
            accEl.textContent = (data.test_metrics.accuracy * 100).toFixed(1) + '%';
        }

        // Update F1 Score
        const f1El = document.getElementById('test-f1');
        if (f1El && data.test_metrics) {
            f1El.textContent = (data.test_metrics.f1_score * 100).toFixed(1) + '%';
        }

        // Update Best K
        const kEl = document.getElementById('best-k');
        if (kEl && data.best_k) {
            kEl.textContent = 'K=' + data.best_k;
        }

        // Update Per-Class Metrics
        const perClassGrid = document.getElementById('per-class-grid');
        if (perClassGrid && data.test_metrics && data.test_metrics.per_class) {
            perClassGrid.innerHTML = '';
            const classColors = {
                'Matang': 'var(--color-matang)',
                'Mengkal': 'var(--color-mengkal)',
                'Mentah': 'var(--color-mentah)',
            };
            for (const [className, metrics] of Object.entries(data.test_metrics.per_class)) {
                const card = document.createElement('div');
                card.className = 'per-class-card';
                card.style.borderLeftColor = classColors[className] || '#94a3b8';
                card.innerHTML = `
                    <h4 style="color: ${classColors[className] || '#94a3b8'}; margin-bottom: 0.5rem;">${className}</h4>
                    <div class="per-class-metrics-row">
                        <div><span class="metric-name">Precision</span><span class="metric-value" style="color: ${classColors[className]};">${(metrics.precision * 100).toFixed(1)}%</span></div>
                        <div><span class="metric-name">Recall</span><span class="metric-value" style="color: ${classColors[className]};">${(metrics.recall * 100).toFixed(1)}%</span></div>
                        <div><span class="metric-name">F1</span><span class="metric-value" style="color: ${classColors[className]};">${(metrics.f1_score * 100).toFixed(1)}%</span></div>
                    </div>
                `;
                perClassGrid.appendChild(card);
            }
        }

        // Update Confusion Matrix
        const cmContainer = document.getElementById('confusion-matrix-container');
        if (cmContainer && data.test_confusion_matrix && data.test_confusion_matrix.length > 0) {
            const labels = ['Matang', 'Mengkal', 'Mentah'];
            const cm = data.test_confusion_matrix;
            
            let html = '<table class="confusion-table"><thead><tr><th></th>';
            labels.forEach(l => html += `<th class="cm-header">${l}</th>`);
            html += '</tr></thead><tbody>';
            
            // Find max value for color intensity
            let maxVal = 0;
            cm.forEach(row => row.forEach(val => { if (val > maxVal) maxVal = val; }));
            
            for (let i = 0; i < cm.length; i++) {
                html += `<tr><td class="cm-row-label">${labels[i]}</td>`;
                for (let j = 0; j < cm[i].length; j++) {
                    const val = cm[i][j];
                    const intensity = maxVal > 0 ? val / maxVal : 0;
                    const isDiag = i === j;
                    const bgColor = isDiag 
                        ? `rgba(16, 185, 129, ${0.15 + intensity * 0.6})` 
                        : (val > 0 ? `rgba(239, 68, 68, ${0.1 + intensity * 0.4})` : 'transparent');
                    html += `<td class="cm-cell" style="background: ${bgColor}; font-weight: ${isDiag ? '700' : '400'};">${val}</td>`;
                }
                html += '</tr>';
            }
            html += '</tbody></table>';
            cmContainer.innerHTML = html;
        }

        // Update Mapping Description
        const listEl = document.getElementById('cluster-mapping-list');
        if (listEl && data.label_mapping) {
            listEl.innerHTML = '';
            for (const [key, val] of Object.entries(data.label_mapping)) {
                const li = document.createElement('li');
                li.innerHTML = `<strong>Label ${key}:</strong> ${val}`;
                listEl.appendChild(li);
            }
        }

        const bgColors = ['rgba(16, 185, 129, 0.8)', 'rgba(245, 158, 11, 0.8)', 'rgba(239, 68, 68, 0.8)'];
        const borderColors = ['#059669', '#d97706', '#dc2626'];

        // Train Distribution Chart
        if (data.train_distribution) {
            const ctxTrain = document.getElementById('trainDistChart').getContext('2d');
            new Chart(ctxTrain, {
                type: 'bar',
                data: {
                    labels: ['Matang', 'Mengkal', 'Mentah'],
                    datasets: [{
                        label: 'Jumlah Gambar Latih',
                        data: [data.train_distribution['0']||0, data.train_distribution['1']||0, data.train_distribution['2']||0],
                        backgroundColor: bgColors,
                        borderColor: borderColors,
                        borderWidth: 1
                    }]
                },
                options: { responsive: true, plugins: { legend: { display: false } } }
            });
        }

        // Test Distribution Chart
        if (data.test_distribution && Object.keys(data.test_distribution).length > 0) {
            const ctxTest = document.getElementById('testDistChart').getContext('2d');
            new Chart(ctxTest, {
                type: 'bar',
                data: {
                    labels: ['Matang', 'Mengkal', 'Mentah'],
                    datasets: [{
                        label: 'Jumlah Prediksi Test',
                        data: [data.test_distribution['0']||0, data.test_distribution['1']||0, data.test_distribution['2']||0],
                        backgroundColor: bgColors,
                        borderColor: borderColors,
                        borderWidth: 1
                    }]
                },
                options: { responsive: true, plugins: { legend: { display: false } } }
            });
        }

        // Scatter Plot (using ground truth labels now)
        if (data.pca_scatter) {
            const ctxScatter = document.getElementById('pcaScatterChart').getContext('2d');
            
            const datasets = [
                { label: 'Matang', data: [], backgroundColor: bgColors[0] },
                { label: 'Mengkal', data: [], backgroundColor: bgColors[1] },
                { label: 'Mentah', data: [], backgroundColor: bgColors[2] }
            ];

            data.pca_scatter.forEach(pt => {
                const labelIdx = pt.label !== undefined ? pt.label : pt.cluster;
                if (datasets[labelIdx]) {
                    datasets[labelIdx].data.push({x: pt.x, y: pt.y});
                }
            });

            new Chart(ctxScatter, {
                type: 'scatter',
                data: { datasets: datasets },
                options: {
                    responsive: true,
                    scales: {
                        x: { title: { display: true, text: 'PCA 1 (Komponen Utama 1)' } },
                        y: { title: { display: true, text: 'PCA 2 (Komponen Utama 2)' } }
                    }
                }
            });
        }

        // Test Predictions List
        if (data.test_predictions && data.test_predictions.length > 0) {
            const predListEl = document.getElementById('test-predictions-list');
            if (predListEl) {
                predListEl.innerHTML = '';
                data.test_predictions.forEach(item => {
                    const li = document.createElement('li');
                    li.style.marginBottom = '0.5rem';
                    li.style.borderBottom = '1px solid #e2e8f0';
                    li.style.paddingBottom = '0.5rem';
                    
                    const predLabel = item.predicted_label !== undefined ? item.predicted_label : item.cluster;
                    const predName = item.predicted_name || item.label;
                    const trueName = item.true_name || '';
                    const confidence = item.confidence !== undefined ? ` (${(item.confidence*100).toFixed(1)}%)` : '';
                    const correct = item.correct !== undefined ? (item.correct ? ' ✅' : ' ❌') : '';
                    const color = borderColors[predLabel] || '#94a3b8';
                    
                    let html = `<strong>${item.filename}</strong> → `;
                    html += `<span style="color: ${color}; font-weight: 600;">${predName}</span>${confidence}${correct}`;
                    if (trueName && item.correct === false) {
                        html += ` <em style="color: #94a3b8;">(asli: ${trueName})</em>`;
                    }
                    li.innerHTML = html;
                    predListEl.appendChild(li);
                });
            }
        }
    })
    .catch(err => console.error("Gagal memuat dashboard:", err));
}

let liveChartInstance = null;

function drawLivePcaChart(px, py, pLabel) {
    fetch('/api/dashboard_data')
    .then(res => res.json())
    .then(data => {
        if (!data.pca_scatter) return;

        const bgColors = ['rgba(16, 185, 129, 0.4)', 'rgba(245, 158, 11, 0.4)', 'rgba(239, 68, 68, 0.4)'];
        
        const datasets = [
            { label: 'Matang', data: [], backgroundColor: bgColors[0], pointRadius: 3, order: 2 },
            { label: 'Mengkal', data: [], backgroundColor: bgColors[1], pointRadius: 3, order: 2 },
            { label: 'Mentah', data: [], backgroundColor: bgColors[2], pointRadius: 3, order: 2 }
        ];

        data.pca_scatter.forEach(pt => {
            const labelIdx = pt.label !== undefined ? pt.label : pt.cluster;
            if (datasets[labelIdx]) {
                datasets[labelIdx].data.push({x: pt.x, y: pt.y});
            }
        });

        // Create a canvas with the star emoji '⭐'
        const starCanvas = document.createElement('canvas');
        starCanvas.width = 32;
        starCanvas.height = 32;
        const starCtx = starCanvas.getContext('2d');
        starCtx.font = '24px serif';
        starCtx.textAlign = 'center';
        starCtx.textBaseline = 'middle';
        starCtx.fillText('⭐', 16, 16);

        // ADD NEW UPLOADED IMAGE POINT (THE STAR)
        datasets.push({
            label: 'Gambar Input',
            data: [{x: px, y: py}],
            pointStyle: starCanvas,
            pointRadius: 16,
            pointHoverRadius: 20,
            order: 1
        });

        const ctx = document.getElementById('livePcaChart').getContext('2d');
        if (liveChartInstance) {
            liveChartInstance.destroy();
        }

        liveChartInstance = new Chart(ctx, {
            type: 'scatter',
            data: { datasets: datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            usePointStyle: true
                        }
                    }
                },
                scales: {
                    x: { title: { display: true, text: 'PCA 1' } },
                    y: { title: { display: true, text: 'PCA 2' } }
                }
            }
        });
    });
}
