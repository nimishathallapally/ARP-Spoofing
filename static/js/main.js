// Auto-hide alerts after 5 seconds
document.addEventListener('DOMContentLoaded', function() {
    const alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
    alerts.forEach(alert => {
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Animate stat counters
    animateStatCounters();
    
    // Add fade-in animation to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        setTimeout(() => {
            card.classList.add('fade-in');
        }, index * 100);
    });
});

// Animate stat counters
function animateStatCounters() {
    const counters = document.querySelectorAll('.counter-value, .stat-value');
    
    counters.forEach(counter => {
        // Store original value
        const originalText = counter.textContent;
        const target = parseFloat(originalText.replace(/[^0-9.]/g, ''));
        
        if (!isNaN(target) && target > 0) {
            let current = 0;
            const increment = target / 50;
            const isPercentage = originalText.includes('%');
            const isDecimal = originalText.includes('.');
            
            const timer = setInterval(() => {
                current += increment;
                if (current >= target) {
                    current = target;
                    clearInterval(timer);
                }
                
                if (isPercentage) {
                    counter.textContent = current.toFixed(1) + '%';
                } else if (isDecimal) {
                    counter.textContent = current.toFixed(1);
                } else {
                    counter.textContent = Math.floor(current).toLocaleString();
                }
            }, 20);
        }
    });
}

// Export results to CSV
function exportToCSV() {
    const table = document.getElementById('detections-table');
    if (!table) {
        showNotification('No data to export', 'warning');
        return;
    }
    
    let csv = [];
    const rows = table.querySelectorAll('tr');
    
    rows.forEach(row => {
        const cols = row.querySelectorAll('td, th');
        const csvRow = [];
        cols.forEach(col => {
            csvRow.push('"' + col.textContent.trim() + '"');
        });
        csv.push(csvRow.join(','));
    });
    
    const csvContent = csv.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'arp_spoofing_detections_' + new Date().toISOString().split('T')[0] + '.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
    
    showNotification('Results exported successfully', 'success');
}

// Print results
function printResults() {
    window.print();
}

// Show notification toast
function showNotification(message, type = 'info') {
    const toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        const container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'position-fixed top-0 end-0 p-3';
        container.style.zIndex = '11';
        document.body.appendChild(container);
    }
    
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');
    
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    document.getElementById('toast-container').appendChild(toast);
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
    
    toast.addEventListener('hidden.bs.toast', () => {
        toast.remove();
    });
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Validate file before upload
function validateFile(file) {
    const maxSize = 50 * 1024 * 1024; // 50MB
    const allowedTypes = ['text/csv', 'application/vnd.ms-excel'];
    
    if (file.size > maxSize) {
        showNotification('File size exceeds 50MB limit', 'danger');
        return false;
    }
    
    if (!allowedTypes.includes(file.type) && !file.name.endsWith('.csv')) {
        showNotification('Only CSV files are allowed', 'danger');
        return false;
    }
    
    return true;
}

// Smooth scroll to element
function smoothScrollTo(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// Copy text to clipboard
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showNotification('Copied to clipboard', 'success');
    }).catch(err => {
        showNotification('Failed to copy', 'danger');
    });
}

// Refresh dashboard data
function refreshDashboard() {
    const currentUrl = window.location.href;
    showNotification('Refreshing dashboard...', 'info');
    
    setTimeout(() => {
        window.location.reload();
    }, 500);
}

// Auto-refresh dashboard every 30 seconds (optional)
let autoRefreshInterval;

function startAutoRefresh(intervalSeconds = 30) {
    stopAutoRefresh(); // Clear any existing interval
    
    autoRefreshInterval = setInterval(() => {
        refreshDashboard();
    }, intervalSeconds * 1000);
    
    showNotification(`Auto-refresh enabled (${intervalSeconds}s)`, 'info');
}

function stopAutoRefresh() {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
        showNotification('Auto-refresh disabled', 'info');
    }
}

// Format timestamp
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString();
}

// Calculate detection rate
function calculateDetectionRate(attacks, total) {
    if (total === 0) return 0;
    return ((attacks / total) * 100).toFixed(2);
}

// Generate random color for charts
function generateChartColor(index) {
    const colors = [
        '#3498db', '#e74c3c', '#27ae60', '#f39c12', '#9b59b6',
        '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#16a085'
    ];
    return colors[index % colors.length];
}

// Create Chart.js default configuration
function getDefaultChartConfig(type) {
    return {
        type: type,
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 15,
                        font: {
                            size: 12
                        }
                    }
                }
            }
        }
    };
}

// Download chart as image
function downloadChart(chartId, filename) {
    const canvas = document.getElementById(chartId);
    if (!canvas) {
        showNotification('Chart not found', 'danger');
        return;
    }
    
    const url = canvas.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = url;
    a.download = filename || 'chart.png';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    showNotification('Chart downloaded', 'success');
}

// Highlight table row on click
document.addEventListener('DOMContentLoaded', function() {
    const tables = document.querySelectorAll('.table tbody tr');
    tables.forEach(row => {
        row.addEventListener('click', function() {
            // Remove highlight from all rows
            tables.forEach(r => r.classList.remove('table-active'));
            // Add highlight to clicked row
            this.classList.add('table-active');
        });
    });
});

// Dark mode toggle (optional feature)
function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
    const isDark = document.body.classList.contains('dark-mode');
    localStorage.setItem('darkMode', isDark);
    showNotification(`Dark mode ${isDark ? 'enabled' : 'disabled'}`, 'info');
}

// Load dark mode preference
document.addEventListener('DOMContentLoaded', function() {
    const darkMode = localStorage.getItem('darkMode') === 'true';
    if (darkMode) {
        document.body.classList.add('dark-mode');
    }
});

// Search/filter table
function filterTable(tableId, searchInputId) {
    const input = document.getElementById(searchInputId);
    const table = document.getElementById(tableId);
    
    if (!input || !table) return;
    
    const filter = input.value.toUpperCase();
    const rows = table.getElementsByTagName('tr');
    
    for (let i = 1; i < rows.length; i++) {
        const cells = rows[i].getElementsByTagName('td');
        let found = false;
        
        for (let j = 0; j < cells.length; j++) {
            if (cells[j].textContent.toUpperCase().indexOf(filter) > -1) {
                found = true;
                break;
            }
        }
        
        rows[i].style.display = found ? '' : 'none';
    }
}

console.log('ARP Spoofing Detection System - Main.js loaded');
