// Constants
const CHART_COLORS = {
    primary: {
        fill: 'rgba(59, 130, 246, 0.1)',  // Màu xanh nhạt cho background
        stroke: 'rgb(59, 130, 246)'        // Màu xanh đậm cho đường
    },
    prediction: {
        fill: 'rgba(16, 185, 129, 0.1)',   // Màu xanh lá nhạt cho background
        stroke: 'rgb(16, 185, 129)'        // Màu xanh lá đậm cho đường
    }
};

let priceChart = null;

// Khởi tạo ứng dụng
$(document).ready(function () {
    initializeCompanySearch();
    initializeCharts();
    updateMarketOverview();
    setupEventListeners();

    // Cập nhật thông tin thị trường mỗi 5 phút
    setInterval(updateMarketOverview, 300000);
});

// Khởi tạo tìm kiếm công ty với Select2
function initializeCompanySearch() {
    $('#companySearch').select2({
        ajax: {
            url: '/api/search_companies',
            dataType: 'json',
            delay: 500,  // Tăng delay để tránh gọi API quá nhiều
            data: function (params) {
                return {
                    query: params.term || ''  // Đảm bảo luôn có giá trị query
                };
            },
            processResults: function (data) {
                return {
                    results: data.map(company => ({
                        id: company.symbol,
                        text: `${company.symbol} - ${company.name}`,
                        company: company
                    }))
                };
            },
            cache: true
        },
        minimumInputLength: 2,
        templateResult: formatCompanyOption,
        templateSelection: formatCompanySelection,
        placeholder: 'Search for a company...',
        allowClear: true,
        width: '100%',
        language: {
            inputTooShort: function() {
                return "Please enter at least 2 characters";
            },
            searching: function() {
                return "Searching...";
            },
            noResults: function() {
                return "No companies found";
            }
        }
    }).on('select2:select', onCompanySelect);
}

// Format hiển thị kết quả tìm kiếm
function formatCompanyOption(company) {
    if (!company.company) return company.text;

    return $(`
        <div class="company-option">
            <div class="company-name">
                <strong>${company.company.symbol}</strong> - ${company.company.name}
            </div>
            <div class="company-details">
                <span class="badge ${company.company.is_trained ? 'bg-success' : 'bg-warning'}">
                    ${company.company.is_trained ? 'Trained' : 'Untrained'}
                </span>
                <span class="text-muted">${company.company.sector || 'N/A'}</span>
                ${company.company.country ? `<span class="text-muted">${company.company.country}</span>` : ''}
            </div>
        </div>
    `);
}

// Format hiển thị công ty đã chọn
function formatCompanySelection(company) {
    if (!company.company) return company.text;
    return `${company.company.symbol} - ${company.company.name}`;
}

// Xử lý khi chọn công ty
function onCompanySelect(e) {
    const symbol = e.target.value;
    if (!symbol) return;

    showLoading();

    // Kiểm tra trạng thái công ty
    fetch(`/api/check_company_status/${symbol}`)
        .then(response => response.json())
        .then(data => {
            updateCompanyInfo(data);

            // Hiển thị controls phù hợp
            if (data.is_trained) {
                $('#predictionControls').show();
                $('#historicalControls').hide();
                updateModelTypeOptions(data.trained_models);
            } else {
                $('#predictionControls').hide();
                $('#historicalControls').show();
            }

            // Load dữ liệu lịch sử
            loadHistoricalData(symbol);
        })
        .catch(error => {
            showError('Error checking company status: ' + error);
        })
        .finally(() => {
            hideLoading();
        });
}

// Cập nhật thông tin công ty
function updateCompanyInfo(info) {
    const marketCap = formatMarketCap(info.market_cap);
    const priceChange = formatNumber(info.price_change, 2);
    const priceChangeClass = info.price_change >= 0 ? 'text-success' : 'text-danger';

    $('#companyInfo').html(`
        <div class="company-header">
            <h3>${info.name} (${info.symbol})</h3>
            <div class="current-price">
                <span class="price">${formatNumber(info.current_price, 2)} ${info.currency}</span>
                <span class="${priceChangeClass}">
                    ${priceChange}%
                    <i class="fas fa-${info.price_change >= 0 ? 'caret-up' : 'caret-down'}"></i>
                </span>
            </div>
        </div>
        <div class="company-details-grid">
            <div class="info-item">
                <span class="label">Sector</span>
                <span class="value">${info.sector || 'N/A'}</span>
            </div>
            <div class="info-item">
                <span class="label">Industry</span>
                <span class="value">${info.industry || 'N/A'}</span>
            </div>
            <div class="info-item">
                <span class="label">Market Cap</span>
                <span class="value">${marketCap}</span>
            </div>
            <div class="info-item">
                <span class="label">Country</span>
                <span class="value">${info.country || 'N/A'}</span>
            </div>
        </div>
    `).show();
}

// Cập nhật options cho model type
function updateModelTypeOptions(trainedModels) {
    const modelSelect = $('#modelType');
    modelSelect.empty();

    if (trainedModels && trainedModels.length > 0) {
        trainedModels.forEach(model => {
            modelSelect.append(new Option(model, model));
        });
        modelSelect.prop('disabled', false);
    } else {
        modelSelect.append(new Option('No trained models available', ''));
        modelSelect.prop('disabled', true);
    }
}

// Khởi tạo biểu đồ
function initializeCharts() {
    const ctx = document.getElementById('priceChart').getContext('2d');

    // Set style cho canvas
    ctx.canvas.style.backgroundColor = 'white';

    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Historical Price',
                    data: [],
                    borderColor: CHART_COLORS.primary.stroke,
                    backgroundColor: CHART_COLORS.primary.fill,
                    borderWidth: 2,
                    pointRadius: 2,
                    pointHoverRadius: 5,
                    pointBackgroundColor: CHART_COLORS.primary.stroke,
                    fill: true,
                    tension: 0.4  // Làm mượt đường
                },
                {
                    label: 'Predicted Price',
                    data: [],
                    borderColor: CHART_COLORS.prediction.stroke,
                    backgroundColor: CHART_COLORS.prediction.fill,
                    borderWidth: 2,
                    pointRadius: 2,
                    pointHoverRadius: 5,
                    pointBackgroundColor: CHART_COLORS.prediction.stroke,
                    fill: true,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 15,
                        font: {
                            family: "'Inter', sans-serif",
                            size: 12
                        }
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(255, 255, 255, 0.9)',
                    titleColor: '#1f2937',
                    bodyColor: '#1f2937',
                    borderColor: '#e5e7eb',
                    borderWidth: 1,
                    padding: 10,
                    bodyFont: {
                        family: "'Inter', sans-serif"
                    },
                    titleFont: {
                        family: "'Inter', sans-serif",
                        weight: 'bold'
                    },
                    callbacks: {
                        label: function (context) {
                            return `${context.dataset.label}: $${context.parsed.y.toFixed(2)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            family: "'Inter', sans-serif",
                            size: 11
                        },
                        maxRotation: 45,
                        minRotation: 45
                    }
                },
                y: {
                    display: true,
                    position: 'left',
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    },
                    ticks: {
                        font: {
                            family: "'Inter', sans-serif",
                            size: 11
                        },
                        callback: function (value) {
                            return '$' + value.toFixed(2);
                        }
                    }
                }
            }
        }
    });
}

// Cập nhật hàm updateChartWithHistorical
function updateChartWithHistorical(data) {
    const historicalDataset = {
        label: 'Historical Price',
        data: data.prices,
        borderColor: CHART_COLORS.primary.stroke,
        backgroundColor: CHART_COLORS.primary.fill,
        borderWidth: 2,
        pointRadius: 2,
        pointHoverRadius: 5,
        pointBackgroundColor: CHART_COLORS.primary.stroke,
        fill: true,
        tension: 0.4
    };

    priceChart.data.labels = data.dates;
    priceChart.data.datasets[0] = historicalDataset;
    priceChart.data.datasets[1].data = Array(data.dates.length).fill(null);
    priceChart.update();
}

// Cập nhật hàm updateChartWithPredictions
function updateChartWithPredictions(data) {
    try {
        // Kết hợp dữ liệu lịch sử và dự đoán
    priceChart.data.labels = [
            ...data.historical.dates,
        ...data.predictions.dates
    ];

        // Dataset cho giá lịch sử
    priceChart.data.datasets[0].data = [
            ...data.historical.values,
        ...Array(data.predictions.dates.length).fill(null)
    ];

        // Dataset cho giá dự đoán
    priceChart.data.datasets[1].data = [
            ...Array(data.historical.dates.length).fill(null),
        ...data.predictions.values
    ];

        // Thêm vùng dự đoán (prediction range)
        const minPred = data.statistics.min_prediction;
        const maxPred = data.statistics.max_prediction;
        const rangeData = Array(data.historical.dates.length).fill(null).concat(
            Array(data.predictions.dates.length).fill([minPred, maxPred])
        );

        // Cập nhật chart options
        priceChart.options.plugins.tooltip = {
            mode: 'index',
            intersect: false,
            callbacks: {
                label: function(context) {
                    if (context.dataset.label === 'Historical Price') {
                        return `Real Price: $${context.parsed.y.toFixed(2)}`;
                    } else if (context.dataset.label === 'Predicted Price') {
                        return `Predicted: $${context.parsed.y.toFixed(2)}`;
                    }
                }
            }
        };

        // Thêm vertical line để phân tách dữ liệu thực tế và dự đoán
        const separationIndex = data.historical.dates.length - 1;
        const separationPlugin = {
            id: 'separationLine',
            beforeDraw: function(chart) {
                if (chart.tooltip._active && chart.tooltip._active.length) {
                    const activePoint = chart.tooltip._active[0];
                    const ctx = chart.ctx;
                    const x = activePoint.element.x;
                    const topY = chart.scales.y.top;
                    const bottomY = chart.scales.y.bottom;

                    ctx.save();
                    ctx.beginPath();
                    ctx.moveTo(x, topY);
                    ctx.lineTo(x, bottomY);
                    ctx.lineWidth = 1;
                    ctx.strokeStyle = '#ff0000';
                    ctx.stroke();
                    ctx.restore();
                }
            }
        };

        // Cập nhật chart
        priceChart.update();

        // Hiển thị thống kê
        updateStatistics(data.statistics);

        // Hiển thị khuyến nghị dựa trên dự đoán
        showRecommendation(data.statistics);

    } catch (error) {
        console.error('Error updating chart:', error);
        showError('Error updating chart with predictions');
    }
}

// Thêm hàm hiển thị khuyến nghị
function showRecommendation(statistics) {
    const priceChange = statistics.price_change_percent;
    let recommendation = '';
    let recommendationClass = '';

    if (priceChange > 5) {
        recommendation = 'Strong Buy';
        recommendationClass = 'text-success';
    } else if (priceChange > 2) {
        recommendation = 'Buy';
        recommendationClass = 'text-success';
    } else if (priceChange < -5) {
        recommendation = 'Strong Sell';
        recommendationClass = 'text-danger';
    } else if (priceChange < -2) {
        recommendation = 'Sell';
        recommendationClass = 'text-danger';
    } else {
        recommendation = 'Hold';
        recommendationClass = 'text-warning';
    }

    $('#recommendation').html(`
        <div class="recommendation-box">
            <h4>Trading Recommendation</h4>
            <p class="recommendation ${recommendationClass}">
                <strong>${recommendation}</strong>
            </p>
            <p>Expected Price Change: 
                <span class="${priceChange >= 0 ? 'text-success' : 'text-danger'}">
                    ${priceChange.toFixed(2)}%
                </span>
            </p>
            <p>Price Range: $${statistics.min_prediction.toFixed(2)} - $${statistics.max_prediction.toFixed(2)}</p>
        </div>
    `).show();
}

// Load dữ liệu lịch sử
function loadHistoricalData(symbol) {
    const days = $('#hist_days').val() || 0;
    const months = $('#hist_months').val() || 1;
    const years = $('#hist_years').val() || 0;

    showLoading();

    fetch(`/api/historical/${symbol}?days=${days}&months=${months}&years=${years}`)
        .then(response => response.json())
        .then(data => {
            updateChartWithHistorical(data);
        })
        .catch(error => {
            showError('Error loading historical data: ' + error);
        })
        .finally(() => {
            hideLoading();
        });
}

// Cập nhật hàm handlePredictionSubmit
function handlePredictionSubmit(event) {
    event.preventDefault();

    const symbol = $('#companySearch').val();
    const modelType = $('#modelType').val();
    const days = parseInt($('#pred_days').val()) || 0;
    const months = parseInt($('#pred_months').val()) || 0;
    const selectedYear = parseInt($('#pred_years').val()) || 0;

    if (!symbol || !modelType) {
        showError('Please select a company and model type');
        return;
    }

    if (selectedYear === 0 && days + months <= 0) {
        showError('Please enter a valid prediction period');
        return;
    }

    // Tính số năm dự đoán
    const currentYear = new Date().getFullYear();
    const years = selectedYear ? selectedYear - currentYear : 0;

    showLoading();

    // Gọi API dự đoán
    fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            company: symbol,
            model_type: modelType,
            days: days,
            months: months,
            years: years
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
        // Cập nhật biểu đồ với dữ liệu dự đoán
            updateChartWithPredictions(data);
        // Cập nhật thống kê
            updateStatistics(data.statistics);
        })
        .catch(error => {
        console.error('Prediction error:', error);
        showError('Prediction error: ' + error.message);
        })
        .finally(() => {
            hideLoading();
        });
}

// Cập nhật thống kê
function updateStatistics(statistics) {
    $('#statistics').html(`
        <div class="statistics-grid">
            <div class="stat-item">
                <span class="label">Current Price</span>
                <span class="value">${formatNumber(statistics.current_price, 2)}</span>
            </div>
            <div class="stat-item">
                <span class="label">Average Prediction</span>
                <span class="value">${formatNumber(statistics.average_prediction, 2)}</span>
            </div>
            <div class="stat-item">
                <span class="label">Price Change</span>
                <span class="value ${statistics.price_change_percent >= 0 ? 'text-success' : 'text-danger'}">
                    ${formatNumber(statistics.price_change_percent, 2)}%
                </span>
            </div>
            <div class="stat-item">
                <span class="label">Range</span>
                <span class="value">
                    ${formatNumber(statistics.min_prediction, 2)} - ${formatNumber(statistics.max_prediction, 2)}
                </span>
            </div>
        </div>
    `).show();
}

// Cập nhật thông tin thị trường
function updateMarketOverview() {
    fetch('/api/market_overview')
        .then(response => response.json())
        .then(data => {
            updateMarketIndices(data.market_indices);
            updateTrainedCompaniesList(data.trained_companies);
        })
        .catch(error => {
            console.error('Error updating market overview:', error);
        });
}

// Cập nhật chỉ số thị trường
function updateMarketIndices(marketData) {
    let html = '<div class="market-indices">';

    for (const [market, indices] of Object.entries(marketData)) {
        html += `<div class="market-section"><h4>${market} Market</h4>`;
        indices.forEach(index => {
            const changeClass = index.change >= 0 ? 'positive' : 'negative';
            html += `
                <div class="market-index ${changeClass}">
                    <span class="index-name">${index.name}</span>
                    <span class="index-price">${formatNumber(index.price, 2)}</span>
                    <span class="index-change ${changeClass}">
                        ${formatNumber(index.change, 2)}%
                        <i class="fas fa-${index.change >= 0 ? 'caret-up' : 'caret-down'}"></i>
                    </span>
                </div>
            `;
        });
        html += '</div>';
    }

    html += '</div>';
    $('#marketOverview').html(html);
}

// Cập nhật danh sách công ty đã train
function updateTrainedCompaniesList(companies) {
    if (!companies || companies.length === 0) {
        $('#trainedCompanies').html('<p>No trained companies available</p>');
        return;
    }

    let html = '<div class="trained-companies-grid">';
    companies.forEach(company => {
        const priceChangeClass = company.price_change >= 0 ? 'text-success' : 'text-danger';
        html += `
            <div class="company-card" data-symbol="${company.symbol}">
                <div class="company-card-header">
                    <h5>${company.symbol}</h5>
                    <span class="${priceChangeClass}">
                        ${formatNumber(company.price_change, 2)}%
                        <i class="fas fa-${company.price_change >= 0 ? 'caret-up' : 'caret-down'}"></i>
                    </span>
                </div>
                <div class="company-card-body">
                    <p class="company-name">${company.name}</p>
                    <p class="company-sector">${company.sector || 'N/A'}</p>
                    <div class="model-tags">
                        ${company.trained_models.map(model =>
            `<span class="model-tag">${model}</span>`
        ).join('')}
                    </div>
                </div>
            </div>
        `;
    });
    html += '</div>';

    $('#trainedCompanies').html(html);
}

// Helper Functions
function formatNumber(number, decimals = 0) {
    if (number === null || number === undefined || isNaN(number)) {
        return 'N/A';
    }
    return Number(number).toFixed(decimals);
}

function formatMarketCap(marketCap) {
    if (!marketCap || marketCap === 'N/A') return 'N/A';

    const billion = 1000000000;
    const million = 1000000;

    if (marketCap >= billion) {
        return `$${(marketCap / billion).toFixed(2)}B`;
    } else if (marketCap >= million) {
        return `$${(marketCap / million).toFixed(2)}M`;
    } else {
        return `$${marketCap.toLocaleString()}`;
    }
}

function showLoading() {
    $('#loadingOverlay').show();
}

function hideLoading() {
    $('#loadingOverlay').hide();
}

function showError(message) {
    $('#alertMessage')
        .removeClass('alert-success')
        .addClass('alert-danger')
        .text(message)
        .fadeIn();

    setTimeout(() => {
        $('#alertMessage').fadeOut();
    }, 5000);
}

function showSuccess(message) {
    $('#alertMessage')
        .removeClass('alert-danger')
        .addClass('alert-success')
        .text(message)
        .fadeIn();

    setTimeout(() => {
        $('#alertMessage').fadeOut();
    }, 3000);
}

// Setup Event Listeners
function setupEventListeners() {
    $('#predictionForm').on('submit', handlePredictionSubmit);
    $('#loadHistoricalBtn').on('click', () => {
        const symbol = $('#companySearch').val();
        if (symbol) {
            loadHistoricalData(symbol);
        }
    });
}